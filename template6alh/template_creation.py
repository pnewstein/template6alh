"""
API functions specificaly for template creations
"""

from datetime import datetime
from subprocess import run
from logging import getLogger
from pathlib import Path

from sqlalchemy.orm import Session, joinedload, aliased
from sqlalchemy import select
import numpy as np
import nrrd
import click

from .sql_classes import Channel, Image, AnalysisStep, ChannelMetadata, GlobalConfig
from .execptions import InvalidStepError, BadImageFolder
from .sql_utils import (
    perform_analysis_step,
    get_path,
    save_channel_to_disk,
    validate_db,
    get_imgs,
    check_progress,
    ConfigDict,
)
from .utils import get_cmtk_executable, FlipLiteral, run_with_logging, get_target_grid
from . import matplotlib_slice

logger = getLogger("template6alh")


def select_images(session: Session, image_paths: list[str] | None, flip: FlipLiteral):
    """
    Selects images for template creation by rendering thoses images in normal coordinates. Uses image_paths and flip to select the right image
    """
    # get the right channels
    if image_paths is None:
        all_images = (
            session.execute(
                select(Image)
                .join(Image.channels)
                .join(Channel.mdata)
                .filter(ChannelMetadata.key == "flip", ChannelMetadata.value == flip)
            )
            .unique()
            .all()
        )
        if not all_images:
            raise InvalidStepError(f"No images with flip {flip}")
        image_paths = [i[0].folder for i in all_images]
    query = (
        select(ChannelMetadata, Channel, AnalysisStep)
        .join(ChannelMetadata.channel)
        .join(Channel.image)
        .join(Channel.producer)
        .options(
            joinedload(ChannelMetadata.channel, Channel.producer),
            joinedload(ChannelMetadata.channel, Channel.image),
        )
        .filter(ChannelMetadata.key == "flip", ChannelMetadata.value == flip)
        .order_by(AnalysisStep.runtime.desc())
    )
    channels: list[Channel] = []
    for image_path in image_paths:
        result = session.execute(query.filter(Image.folder == image_path)).all()
        if not result:
            raise BadImageFolder(image_path)
        channels.append(result[0][1])
    # copy each channel data
    for input_channel in channels:
        step = AnalysisStep()
        step.function = "select_images"
        step.kwargs = repr({"flip": flip})
        step.runtime = datetime.now()
        output_channel = Channel()
        output_channel.path = "for_template.nrrd"
        output_channel.channel_type = "mask"
        perform_analysis_step(
            session, step, [input_channel], [output_channel], 1, copy_scale=True
        )
        # being flipped, we want the abs
        output_channel.scalex = abs(output_channel.scalex)
        output_channel.scaley = abs(output_channel.scaley)
        output_channel.scalez = abs(output_channel.scalez)
        assert output_channel.id is not None
        data, _ = nrrd.read(str(get_path(session, input_channel)))
        (flip_axs,) = np.where([int(v) for v in flip])
        fliped_data = np.flip(data, flip_axs)
        save_channel_to_disk(session, output_channel, fliped_data)
    session.commit()


def iterative_mask_template(session: Session, make_template=True):
    """
    takes previously selected templates and makes a groupwise template using cmtk
    """
    validate_db(session)
    stmt = (
        select(AnalysisStep, Channel, Image)
        .join(AnalysisStep.output_channels)
        .join(Channel.image)
        .filter(AnalysisStep.function == "select_images")
        .order_by(AnalysisStep.runtime.desc())
    )
    results = session.execute(stmt).all()
    visited_image_ids: set[int] = set()
    channels: list[Channel] = []
    # make sure only the first of each image is added
    for _, channel, image in results:
        if image.id not in visited_image_ids:
            visited_image_ids.add(image.id)
            channels.append(channel)
    # make calls to cmtk
    sh_args = [get_cmtk_executable("iterative_shape_averaging")] + [
        get_path(session, c) for c in channels
    ]
    cwd = get_path(session, None)
    template_dir = cwd / "template"
    template_dir.mkdir(exist_ok=True)
    if make_template:
        result = run(sh_args, capture_output=True, check=False, cwd=cwd)
        logger.info(result.stdout.decode())
        logger.debug(result.stderr.decode())
        result.check_returncode()
        paths = list((cwd / "isa/pass5").glob("*for_template.nii.gz"))
        assert len(paths) != 0
        sh_args = [
            get_cmtk_executable("average_images"),
            "--outfile-name",
            "template/mask_template.nrrd",
        ] + paths
        result = run(sh_args, capture_output=True, check=False, cwd=cwd)
        logger.info(result.stdout.decode())
        logger.debug(result.stderr.decode())
        result.check_returncode()
    # postprocesses template
    mask_template_path = template_dir / "mask_template.nrrd"
    template, template_md = nrrd.read(str(mask_template_path))
    template = template * (254 / template.max())
    template = template.astype(np.uint8)
    nrrd.write(file=str(mask_template_path), data=template, header=template_md)
    # add to database
    ConfigDict(session)["mask_template_path"] = str(mask_template_path.resolve())
    session.commit()


def reformat_fasii(session: Session, image_paths: list[str] | None):
    """
    takes all of the raw channels and warp masks and reformats the fasII chan
    accoring to that mask
    """
    images = get_imgs(session, image_paths)
    warp_fasiis: list[tuple[Channel, Channel]] = []
    for image in images:
        neuropil_chan_number = image.raw_file.fasii_chan
        assert neuropil_chan_number is not None
        WarpXform = aliased(Channel)
        AlignToMask = aliased(AnalysisStep)
        FasiiRaw = aliased(Channel)
        results = session.execute(
            select(AlignToMask, WarpXform, FasiiRaw)
            .filter(AlignToMask.function == "align-to-mask")
            .join(WarpXform, AlignToMask.output_channels)
            .filter(WarpXform.channel_type == "xform")
            .join(Image, WarpXform.image)
            .filter(Image.folder == image.folder)
            .join(FasiiRaw, Image.channels)
            .filter(FasiiRaw.number == neuropil_chan_number)
            .filter(FasiiRaw.channel_type == "raw")
            .order_by(AlignToMask.runtime.desc())
        ).all()
        if len(results) == 0:
            logger.warning("No results for image %s", image.folder)
            continue
        all_raw_chans = [r[2] for r in results]
        assert len(set(all_raw_chans)) == len(all_raw_chans)
        _, warp, fasii = results[0]
        warp_fasiis.append((warp, fasii))
    if len(warp_fasiis) == 0:
        raise InvalidStepError("No images to process")
    for input_channels in warp_fasiis:
        check_progress(session, input_channels, 3)
    for warp, fasii in warp_fasiis:
        target_grid = get_target_grid(get_path(session, fasii))
        step = AnalysisStep()
        step.function = "reformat-fasii"
        step.kwargs = "{}"
        step.runtime = datetime.now()
        warp_aligned = Channel()
        warp_aligned.path = "mask_warped_fasii.nrrd"
        chan_number = fasii.image.raw_file.fasii_chan
        assert chan_number is not None
        warp_aligned.number = chan_number
        warp_aligned.channel_type = "image"
        perform_analysis_step(
            session, step, [warp, fasii], [warp_aligned], 3, copy_scale=True
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
                get_path(session, warp),
            )
        )
    session.commit()


def write_landmarks(session: Session):
    config_dict = ConfigDict(session)
    in_path = Path(config_dict["mask_template_path"])
    if not in_path.exists():
        raise InvalidStepError("Missing template at %s", str(in_path))
    out_path = in_path.with_suffix(".landmarks")
    config_dict["mask_template_landmarks_path"] = str(out_path)
    session.commit()
    slicer = matplotlib_slice.write_landmarks(in_path, out_path)
    click.confirm("Close all windows?")
    slicer.quit()
