"""
defines high level functions to manipulate the data

there is a one to one correspondence between api functions and cli options
"""

from pathlib import Path
from typing import Any
from datetime import datetime
import logging

from sqlalchemy.orm import Session, joinedload, aliased
from sqlalchemy import select, Engine, inspect
import nrrd
import click
from scipy import ndimage as ndi
import numpy as np

from .segment_neuropil import make_neuropil_mask, default_args_make_neuropil_mask
from .read_raw_data import read_czi, read_test
from .sql_classes import (
    Channel,
    ChannelMetadata,
    AnalysisStep,
    Base,
    RawFile,
    GlobalConfig,
    Image,
)
from .sql_utils import (
    get_imgs,
    get_path,
    perform_analysis_step,
    check_progress,
    validate_db,
    save_channel_to_disk,
    get_mask_template_path,
    ConfigDict,
    select_most_recent,
    select_recent_landmark_xform_and_mask,
)
from .execptions import NoRawData, InvalidStepError
from .utils import (
    validate_channels,
    get_db_path,
    get_logfile_path,
    get_cmtk_executable,
    run_with_logging,
    get_landmark_affine,
    get_target_grid,
)
from .matplotlib_slice import ImageSlicer, write_landmarks

logger = logging.getLogger(__name__)


def get_paths(session: Session, db_path: Path | None) -> dict[str, Path]:
    """
    gets relivent paths
    """
    if db_path is None:
        db_path = get_db_path()
    out_dict = {"database path": db_path, "logfile path": get_logfile_path()}
    assert session.bind is not None
    if GlobalConfig.__tablename__ in inspect(session.bind).get_table_names():
        out_dict["Data cache prefix"] = Path(ConfigDict(session)["prefix_dir"])
    return out_dict


def init(
    engine: Engine,
    raw_data: list[Path],
    root_dir: Path,
    neuropil_chan: int | None,
    fasii_chan: int | None,
    eve_chan: int | None,
):
    """
    reads the raw files caching each channel as an nrrd and writes a db for all
    of the raw files, adding raw_files, images, channels, global config

    raises NoRawData, ChannelValidationError
    """
    validate_channels([neuropil_chan, fasii_chan, eve_chan])
    root_dir.mkdir(exist_ok=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        ConfigDict(session)["prefix_dir"] = str(root_dir)
        add_more_raw(
            session=session,
            raw_data=raw_data,
            neuropil_chan=neuropil_chan,
            fasii_chan=fasii_chan,
            eve_chan=eve_chan,
        )
        session.commit()
        if session.query(RawFile).count() == 0:
            raise NoRawData(raw_data)


def add_more_raw(
    session: Session,
    raw_data: list[Path],
    neuropil_chan: int | None,
    fasii_chan: int | None,
    eve_chan: int | None,
):
    """
    adds more images from more raw data files


    raises NoRawData, ChannelValidationError
    """
    validate_db(session)
    validate_channels([neuropil_chan, fasii_chan, eve_chan])
    for path in raw_data:
        if path.suffix == ".czi":
            reader = read_czi(path)
        elif path == Path("test"):
            reader = read_test(path)
        else:
            logger.warning("Failed to read %s", path)
            continue
        rf = RawFile()
        rf.path = str(path)
        rf.neuropil_chan = neuropil_chan
        rf.fasii_chan = fasii_chan
        rf.eve_chan = eve_chan
        session.add(rf)
        for metadata, image_array in reader:
            logger.info("read file from %s", path)
            image = Image(folder="", progress=0, raw_file=rf)
            # fix folder name
            session.add(image)
            session.flush()
            assert image.id is not None
            image.folder = f"{image.id:03d}"
            get_path(session, image).mkdir(parents=True, exist_ok=True)
            for i, channel_array in enumerate(image_array, 1):
                channel = Channel()
                channel.path = f"chan{i}.nrrd"
                channel.step_number = 0
                channel.channel_type = "raw"
                channel.number = i
                channel.scalex = metadata["X"]
                channel.scaley = metadata["Y"]
                channel.scalez = metadata["Z"]
                channel.image = image
                session.add(channel)
                save_channel_to_disk(session, channel, channel_array)
    session.commit()


def segment_neuropil(
    session: Session,
    image_paths: list[str] | None,
    new_scale: float | None,
    filter_sigma: float | None,
    opening_size: float | None,
):
    """
    segments the neuropil all of the images with the ids

    new_scale: float: the output scale in um. Default is 1. negative numbers
        are interpreted as no new scale
    filter_sigma: float, a sigma for a noise removal filter: default .1 um
    opening_size: float, the size where smaller objects will be removed. default 1um

    Raises
        BadImageFolder
        BadInputImages
        SkippingStep
    """
    validate_db(session)
    kwargs: dict["str", Any] = {
        "new_scale": (
            new_scale
            if new_scale is not None
            else default_args_make_neuropil_mask["new_scale"]
        ),
        "filter_sigma": (
            filter_sigma
            if filter_sigma is not None
            else default_args_make_neuropil_mask["filter_sigma"]
        ),
        "opening_size": (
            opening_size
            if opening_size is not None
            else default_args_make_neuropil_mask["opening_size"]
        ),
    }
    if kwargs["new_scale"] < 0:
        kwargs["new_scale"] = None
    images = get_imgs(session, image_paths)
    for image in images:
        neuropil_chan_number = image.raw_file.neuropil_chan
        assert neuropil_chan_number is not None
        select_raw_channel = (
            select(Channel)
            .where(Channel.image_id == image.id)
            .where(Channel.number == neuropil_chan_number)
            .where(Channel.channel_type == "raw")
        )
        raw_channel = session.execute(select_raw_channel).scalar_one()
        _ = check_progress(session, [raw_channel], 1)
        in_data, _ = nrrd.read(str(get_path(session, raw_channel)))
        old_scale = (raw_channel.scalez, raw_channel.scaley, raw_channel.scalex)
        out_data = make_neuropil_mask(in_data, old_scale, **kwargs)
        if kwargs["new_scale"] is None:
            out_scale = old_scale
        else:
            out_scale = tuple([kwargs["new_scale"]] * 3)
        # write out data
        out_channel = Channel()
        out_channel.path = f"neuropil_mask.nrrd"
        out_channel.channel_type = "mask"
        out_channel.scalez = out_scale[0]
        out_channel.scaley = out_scale[1]
        out_channel.scalex = out_scale[2]
        # end iterate flip
        perform_analysis_step(
            session,
            AnalysisStep(
                function="make-neuropil-mask",
                kwargs=repr(kwargs),
                runtime=datetime.now(),
            ),
            [raw_channel],
            [out_channel],
            1,
        )
        # write out all of the channels
        assert out_channel.id is not None
        save_channel_to_disk(session, out_channel, out_data)
        session.commit()


def select_neuropil_fasii(session: Session, image_paths: list[str] | None):
    """
    uses a gui to get landmars from an image
    """
    validate_db(session)
    images = get_imgs(session, image_paths)
    FasiiRaw = aliased(Channel)
    masks_fasiis: list[tuple[Channel, Channel]] = []
    for image in images:
        neuropil_chan_number = image.raw_file.fasii_chan
        channels_or_none = session.execute(
            select_most_recent("make-neuropil-mask", image, select(Channel, FasiiRaw))
            .join(FasiiRaw, Image.channels)
            .filter(FasiiRaw.number == neuropil_chan_number)
            .filter(FasiiRaw.channel_type == "raw")
        ).first()
        if channels_or_none is None:
            logger.warning("no channel for image %s", image.folder)
            continue
        mask, fasii = channels_or_none
        assert mask.channel_type == "mask"
        assert fasii.channel_type == "raw"
        masks_fasiis.append((mask, fasii))
    if len(masks_fasiis) == 0:
        raise InvalidStepError("No images found")
    for input_channels in masks_fasiis:
        check_progress(session, input_channels, 1)
    for mask, fasii in masks_fasiis:
        step = AnalysisStep()
        step.function = "select-neuropil-fasii"
        step.kwargs = "{}"
        step.runtime = datetime.now()
        masked_fasii = Channel()
        masked_fasii.path = "neuropil_fasii.nrrd"
        chan_number = fasii.image.raw_file.fasii_chan
        assert chan_number is not None
        masked_fasii.number = chan_number
        masked_fasii.channel_type = "image"
        perform_analysis_step(
            session, step, [mask, fasii], [masked_fasii], 1, copy_scale=True
        )
        mask_data, _ = nrrd.read(str(get_path(session, mask)))
        fasii_data, fasii_md = nrrd.read(str(get_path(session, fasii)))
        scale_frac = np.array(fasii_data.shape) / np.array(mask_data.shape)
        # get_spacings(mask_md) / get_spacings(fasii_md)
        rescaled_mask_data = ndi.zoom(mask_data, scale_frac, order=0)
        assert rescaled_mask_data.shape == fasii_data.shape, "resize failed"
        # out_data = fasii_data * rescaled_mask_data
        out_data = fasii_data.copy()
        out_data[~rescaled_mask_data.astype(bool)] = 0
        nrrd.write(str(get_path(session, masked_fasii)), out_data, header=fasii_md)
    session.commit()


def clean(session: Session):
    """
    removes all files from database which are no longer on the disk
    """
    validate_db(session)
    channels = (
        session.execute(select(Channel).options(joinedload(Channel.image)))
        .scalars()
        .all()
    )
    for channel in channels:
        if get_path(session, channel).exists():
            continue
        for cm in channel.mdata:
            session.delete(cm)
        logger.info("removing %s from db", channel.path)
        session.delete(channel)
    session.commit()


def make_landmarks(session: Session, image_paths: list[str] | None, skip=False):
    """
    uses a gui to get landmars from an image
    """
    validate_db(session)
    images = get_imgs(session, image_paths)
    channels: list[Channel] = []
    for image in images:
        channel_or_none = (
            session.execute(select_most_recent("make-neuropil-mask", image))
            .scalars()
            .first()
        )
        if channel_or_none is None:
            logger.warning("no channel for image %s", image.folder)
            continue
        channels.append(channel_or_none)
    if len(channels) == 0:
        raise InvalidStepError("No images found")
    out_channels: list[Channel] = []
    for channel in channels:
        out_channel = Channel()
        out_channel.channel_type = "landmarks"
        out_channel.path = ".landmarks"
        step = AnalysisStep()
        step.function = "make-landmarks"
        step.kwargs = "{}"
        step.runtime = datetime.now()
        perform_analysis_step(
            session, step, [channel], [out_channel], 2, copy_scale=True
        )
        out_channels.append(out_channel)
    session.commit()
    slicers: list[ImageSlicer] = []
    if not skip:
        for channel, out_channel in zip(channels, out_channels):
            slicers.append(
                write_landmarks(
                    get_path(session, channel), get_path(session, out_channel)
                )
            )
        click.confirm("Close all images")
        for slicer in slicers:
            slicer.quit()


def landmark_register(session: Session, image_paths: list[str] | None):
    """
    performs a landmark registration
    """
    validate_db(session)
    images = get_imgs(session, image_paths)
    _, template_landmarks = get_mask_template_path(session)
    channels: list[Channel] = []
    for image in images:
        channel_or_none = (
            session.execute(select_most_recent("make-landmarks", image))
            .scalars()
            .first()
        )
        if channel_or_none is None:
            logger.warning("no channel for image %s", image.folder)
            continue
        channels.append(channel_or_none)
    if len(channels) == 0:
        raise InvalidStepError("No images found")
    for landmark in channels:
        check_progress(session, [landmark], 3)
    for landmark in channels:
        xform = Channel()
        xform.channel_type = "xform"
        xform.path = "landmark.xform"
        step = AnalysisStep()
        step.function = "landmark-register"
        step.kwargs = "{}"
        step.runtime = datetime.now()
        perform_analysis_step(session, step, [landmark], [xform], 3, copy_scale=True)
        get_landmark_affine(
            get_path(session, landmark),
            template_landmarks,
            get_path(session, xform),
        )
    session.commit()


# TODO allow kwargs
def mask_register(session: Session, image_paths: list[str] | None):
    """
    registers all of the templates masks also reformat fasii image
    """
    validate_db(session)
    template_path, _ = get_mask_template_path(session)
    images = get_imgs(session, image_paths)
    xforms_masks_fasiis: list[tuple[Channel, Channel, Channel]] = []
    for image in images:
        xform_mask = session.execute(
            select_recent_landmark_xform_and_mask(image)
        ).first()
        fasii = (
            session.execute(select_most_recent("select-neuropil-fasii", image))
            .scalars()
            .first()
        )
        if xform_mask is None or fasii is None:
            logger.warning("could find image %s", image.folder)
            continue
        xform, mask = xform_mask
        assert mask.channel_type == "mask"
        assert xform.channel_type == "xform"
        assert fasii.channel_type == "image"
        xforms_masks_fasiis.append((xform, mask, fasii))
    # allert to errors first
    for input_channels in xforms_masks_fasiis:
        _ = check_progress(session, input_channels, 4)
    # do two phase registration
    for xform, mask, fasii in xforms_masks_fasiis:
        # database update
        analysis_step = AnalysisStep()
        analysis_step.function = "mask-register"
        analysis_step.kwargs = "{}"
        analysis_step.runtime = datetime.now()
        affine_xform_chan = Channel()
        affine_xform_chan.path = f"affine_mask.xform"
        affine_xform_chan.channel_type = "xform"
        affine_xform_chan.mdata = [ChannelMetadata(key="xform-type", value="affine")]
        warp_xform_chan = Channel()
        warp_xform_chan.path = f"warp_mask.xform"
        warp_xform_chan.channel_type = "xform"
        warp_xform_chan.mdata = [ChannelMetadata(key="xform-type", value="warp")]
        warp_aligned = Channel()
        warp_aligned.path = "mask_warped_fasii.nrrd"
        chan_number = fasii.image.raw_file.fasii_chan
        assert chan_number is not None
        warp_aligned.number = chan_number
        warp_aligned.channel_type = "aligned"
        perform_analysis_step(
            session,
            analysis_step,
            [xform, mask, fasii],
            [affine_xform_chan, warp_xform_chan, warp_aligned],
            4,
            copy_scale=True,
        )
        target_grid = get_target_grid(get_path(session, fasii), template_path)
        # Call cmtk
        run_with_logging(
            (
                get_cmtk_executable("registration"),
                "--initial",
                get_path(session, xform),
                "--dofs",
                "6,9",
                "--auto-multi-levels",
                "2",
                "-a",
                "0.5",
                "-o",
                get_path(session, affine_xform_chan),
                template_path,
                get_path(session, mask),
            )
        )
        run_with_logging(
            (
                get_cmtk_executable("warp"),
                "--outlist",
                get_path(session, warp_xform_chan),
                "--grid-spacing",
                "80",
                "--fast",
                "--exploration",
                "26",
                "--coarsest",
                "8",
                "--accuracy",
                "0.8",
                "--refine",
                "4",
                "--energy-weight",
                "1e-1",
                "--ic-weight",
                "0",
                "--initial",
                get_path(session, affine_xform_chan),
                template_path,
                get_path(session, mask),
            )
        )
        run_with_logging(
            (
                get_cmtk_executable("reformatx"),
                "-o",
                get_path(session, warp_aligned),
                "--target-grid",
                target_grid,
                "--linear",
                "--floating",
                get_path(session, fasii),
                get_path(session, warp_xform_chan),
            )
        )
    session.commit()
