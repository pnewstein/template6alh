"""
defines high level functions to manipulate the data

there is a one to one corrispondance between api functions and cli options
"""

from pathlib import Path
from typing import Any
from datetime import datetime
import logging
import tempfile

from sqlalchemy.orm import Session, joinedload, aliased
from sqlalchemy import select, Engine, inspect
import nrrd
import numpy as np
import click

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
)
from .execptions import NoRawData, InvalidStepError
from .utils import (
    validate_channels,
    get_db_path,
    get_logfile_path,
    get_cmtk_executable,
    run_with_logging,
    FlipLiteral,
)
from .matplotlib_slice import get_slicer, ImageSlicer

logger = logging.getLogger()


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
        session.delete(channel)
    session.commit()


# TODO allow kwargs
def mask_affine(session: Session, image_paths: list[str] | None):
    """
    alignes all of the templates masks
    """
    validate_db(session)
    template_path = get_mask_template_path(session)
    images = get_imgs(session, image_paths)
    channels_flips: list[tuple[Channel, FlipLiteral]] = []
    for image in images:
        # if image.progress < 1:
        # raise SkippingStep(3, image.progress)
        channels_cmetadata = session.execute(
            select(Channel, ChannelMetadata)
            .join(ChannelMetadata, Channel.mdata)
            .join(AnalysisStep, Channel.producer)
            .join(Image, Channel.image)
            .filter(Image.id == image.id)
            .filter(AnalysisStep.function == "make-neuropil-mask")
            .filter(ChannelMetadata.key == "flip")
            .order_by(AnalysisStep.runtime.desc())
        ).all()
        # get flips
        visteded_flips: set[str] = set()
        for channel, chan_mdata in channels_cmetadata:
            if chan_mdata.value in visteded_flips:
                continue
            visteded_flips.add(chan_mdata.value)
            channels_flips.append((channel, chan_mdata.value))
    assert len(set(channels_flips)) == len(channels_flips)
    # allert to errors first
    for channel, _ in channels_flips:
        _ = check_progress(session, [channel], 2)
    image_map_reformated: dict[Image, list[Channel]] = {i: [] for i in images}
    for channel, flip in channels_flips:
        # get no_flip channel:
        sister_channels = channel.producer.output_channels
        no_flip_channels = [
            c
            for c in sister_channels
            if "000" == next(m.value for m in c.mdata if m.key == "flip")
        ]
        input_path = get_path(session, channel)
        if len(no_flip_channels) != 1:
            raise InvalidStepError(f"cannot find zero flip for {input_path}")
        # figure out inital xform
        analysis_step = AnalysisStep()
        analysis_step.function = "mask-affine"
        analysis_step.kwargs = "{}"
        analysis_step.runtime = datetime.now()
        affine_xform_chan = Channel()
        affine_xform_chan.path = f"affine_{flip}.xform"
        affine_xform_chan.channel_type = "xform"
        affine_xform_chan.mdata = [ChannelMetadata(key="flip", value=flip)]
        affine_aligned_chan = Channel()
        affine_aligned_chan.path = f"reformat_{flip}.nrrd"
        affine_aligned_chan.channel_type = "mask"
        affine_aligned_chan.mdata = [ChannelMetadata(key="flip", value=flip)]
        perform_analysis_step(
            session,
            analysis_step,
            [channel],
            [affine_xform_chan, affine_aligned_chan],
            2,
            copy_scale=True,
        )
        with tempfile.TemporaryDirectory() as folder_str:
            folder = Path(folder_str)
            unfliped_path = get_path(session, no_flip_channels[0])
            affine_xform_path = get_path(session, affine_xform_chan)
            init_xform_path = folder / "init.xform"
            run_with_logging(
                (
                    get_cmtk_executable("make_initial_affine"),
                    "--centers-of-mass",
                    template_path,
                    input_path,
                    init_xform_path,
                )
            )
            run_with_logging(
                (
                    get_cmtk_executable("registration"),
                    "--initial",
                    init_xform_path,
                    "--dofs",
                    "6,9",
                    "--auto-multi-levels",
                    "2",
                    # "-a",
                    # "0.5",
                    "-o",
                    affine_xform_path,
                    template_path,
                    unfliped_path,
                )
            )
        # Now register all images using affine
        run_with_logging(
            (
                get_cmtk_executable("reformatx"),
                "-o",
                get_path(session, affine_aligned_chan),
                "--nn",
                "--floating",
                unfliped_path,
                template_path,
                affine_xform_path,
            )
        )
        image_map_reformated[affine_aligned_chan.image].append(affine_aligned_chan)
    # evaluate xform quality
    template, _ = nrrd.read(str(template_path))
    template_mask = template == 254
    max_score = 2 * template_mask.sum()
    for image, channels in image_map_reformated.items():
        channel_score: list[tuple[Channel, float]] = []
        for channel in channels:
            reformated, _ = nrrd.read(str(get_path(session, channel)))
            sum_template_covered = (reformated[template_mask] > 0).sum()
            sum_reformat_covered = (template[reformated > 0] > 128).sum()
            assert not np.isnan(sum_reformat_covered)
            assert not np.isnan(sum_template_covered)
            channel.mdata.append(
                ChannelMetadata(key="template-covered", value=str(sum_template_covered))
            )
            channel.mdata.append(
                ChannelMetadata(key="reformat-covered", value=str(sum_reformat_covered))
            )
            score = (sum_template_covered + sum_reformat_covered) / max_score
            channel.mdata.append(ChannelMetadata(key="score", value=str(score)))
            channel_score.append((channel, score))
        channel_score.sort(key=lambda e: e[1], reverse=True)
        channel_score_repr = [(c.path, s) for c, s in channel_score]
        logger.debug(f"scores are {channel_score_repr} ")
        if channel_score[1][1] * 1.2 > channel_score[0][1]:
            logger.warning(
                f"Second best flip ({channel_score_repr[1][0]}) is "
                f"quite close to best ({channel_score_repr[0][0]}) "
            )
        for i, (channel, _) in enumerate(channel_score):
            if i == 0:
                channel.mdata.append(ChannelMetadata(key="best-flip", value="yes"))
            else:
                channel.mdata.append(ChannelMetadata(key="best-flip", value="no"))
    session.commit()


def visualize_best_match(session, image_paths: list[str] | None):
    """
    visualizes the best match for each image
    """
    validate_db(session)
    images = get_imgs(session, image_paths)
    channels: list[Channel] = []
    for image in images:
        channel = (
            session.execute(
                select(Channel)
                .join(ChannelMetadata, Channel.mdata)
                .filter(
                    ChannelMetadata.key == "best-flip", ChannelMetadata.value == "yes"
                )
                .join(Image, Channel.image)
                .filter(Image.folder == image.folder)
                .join(AnalysisStep, Channel.producer)
                .order_by(AnalysisStep.runtime.desc())
            )
            .scalars()
            .first()
        )
        if channel is None:
            logger.warning("could not find a best flip for image %s", image.folder)
        channels.append(channel)
    slicers: list[ImageSlicer] = []
    for channel in channels:
        data, _ = nrrd.read(str(get_path(session, channel)))
        slicers.append(get_slicer(data, channel.image.folder))
    click.confirm("Close all windows?")
    for slicer in slicers:
        slicer.quit()


def align_to_mask(session: Session, image_paths: list[str] | None):
    """
    Align the image to the mask template
    """
    validate_db(session)
    template_path = get_mask_template_path(session)
    xform_mask: list[tuple[Channel, Channel]] = []
    images = get_imgs(session, image_paths)
    for image in images:
        unflip_producer = aliased(AnalysisStep)
        xform_chan = aliased(Channel)
        best_mask = aliased(Channel)
        unflip_mask = aliased(Channel)
        unflip_mask_mdata = aliased(ChannelMetadata)
        chan_xform_unflip = session.execute(
            select(Channel, xform_chan, unflip_mask)
            .join(ChannelMetadata, Channel.mdata)
            .filter(ChannelMetadata.key == "best-flip", ChannelMetadata.value == "yes")
            .join(Image, Channel.image)
            .filter(Image.folder == image.folder)
            .join(AnalysisStep, Channel.producer)
            .join(xform_chan, AnalysisStep.output_channels)
            .filter(xform_chan.channel_type == "xform")
            .join(best_mask, AnalysisStep.input_channels)
            .join(unflip_producer, best_mask.producer)
            .join(unflip_mask, unflip_producer.output_channels)
            .join(unflip_mask_mdata, unflip_mask.mdata)
            .filter(unflip_mask_mdata.key == "flip", unflip_mask_mdata.value == "000")
            .order_by(AnalysisStep.runtime.desc())
        ).all()
        n_hits = len(chan_xform_unflip)
        if n_hits == 0:
            raise InvalidStepError("missing channels")
        logger.info("found %s matches for image %s", n_hits, image.folder)
        _, xform, mask = chan_xform_unflip[0]
        xform_mask.append((xform, mask))
    for input_channels in xform_mask:
        check_progress(session, input_channels, 3)
    for xform, mask in xform_mask:
        logger.warning(mask.image.folder)
        analysis_step = AnalysisStep()
        analysis_step.function = "align-to-mask"
        analysis_step.kwargs = "{}"
        analysis_step.runtime = datetime.now()
        warp_mask_xform_chan = Channel()
        warp_mask_xform_chan.path = "warp_mask.xform"
        warp_mask_xform_chan.channel_type = "xform"
        perform_analysis_step(
            session,
            analysis_step,
            [xform, mask],
            [warp_mask_xform_chan],
            3,
            copy_scale=True,
        )
        run_with_logging(
            (
                get_cmtk_executable("warp"),
                "--outlist",
                get_path(session, warp_mask_xform_chan),
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
                get_path(session, xform),
                template_path,
                get_path(session, mask),
            )
        )
    session.commit()
