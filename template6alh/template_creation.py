"""
API functions specificaly for template creations
"""

from typing import Literal
from datetime import datetime
from subprocess import run
from logging import getLogger
from pathlib import Path

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select
import numpy as np
import nrrd

from .sql_classes import Channel, Image, AnalysisStep, ChannelMetadata, GlobalConfig
from .execptions import InvalidStepError, BadImageFolder
from .sql_utils import (
    perform_analysis_step,
    get_path,
    save_channel_to_disk,
    validate_db,
)
from .utils import get_cmtk_executable, FlipLiteral

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
    existing_record = session.execute(
        select(GlobalConfig).filter(GlobalConfig.key == "mask_template_path")
    ).scalar_one_or_none()
    if existing_record is None:
        mask_path_record = GlobalConfig()
        mask_path_record.key = "mask_template_path"
        mask_path_record.value = str(mask_template_path.resolve())
        session.add(mask_path_record)
    else:
        existing_record.value = str(mask_template_path.resolve())
    session.commit()
