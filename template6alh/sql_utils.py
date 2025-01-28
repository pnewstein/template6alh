"""
Some utilities for accessing sql
"""

from typing import Sequence, Literal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, inspect
from sqlalchemy.exc import NoResultFound
import numpy as np
import nrrd

from .sql_classes import Image, GlobalConfig, AnalysisStep, Channel
from .execptions import (
    BadImageFolder,
    BadInputImages,
    SkippingStep,
    UninitializedDatabase,
    CannotFindTemplate,
)

logger = logging.getLogger("template6alh")


@dataclass(frozen=True, slots=True)
class LogEntry:
    """
    Information about an analyis step
    """

    timestamp: datetime
    input_paths: tuple[Path, ...]
    output_paths: tuple[Path, ...]
    function_name: str
    function_kwargs: str
    step_number: int

    @classmethod
    def from_step(cls, session: Session, step: AnalysisStep):
        """
        creates a log entry from a step
        """
        session.add(step)
        return cls(
            timestamp=step.runtime,
            input_paths=tuple(get_path(session, c) for c in step.input_channels),
            output_paths=tuple(get_path(session, c) for c in step.output_channels),
            function_name=step.function,
            function_kwargs=step.kwargs,
            step_number=step.output_channels[0].step_number,
        )

    def format(self) -> str:
        """
        formats into a string
        """
        input_paths = ", ".join(str(p) for p in self.input_paths)
        output_paths = ", ".join(str(p) for p in self.output_paths)
        return (
            f"\t{self.timestamp} [{input_paths}] -- {self.function_name}"
            f"({self.function_kwargs}) --> [{output_paths}]"
        )


def get_log(session: Session, channel: Channel) -> list[LogEntry]:
    """
    Get all of the steps and input images used to create
    """
    session.add(channel)
    # recursivly parent analysis step
    steps: list[AnalysisStep] = []

    def get_parents(channel: Channel):
        # base case
        if channel.producer is None:
            return
        steps.append(channel.producer)
        for in_chan in channel.producer.input_channels:
            return get_parents(in_chan)

    get_parents(channel)
    # convert to logs and sort
    logs = set(LogEntry.from_step(session, step) for step in steps)
    logs = sorted(logs, key=lambda a: a.timestamp)
    return logs


def get_imgs(session: Session, image_ids: list[str] | None) -> Sequence[Image]:
    """
    Gets images from with image_id

    if image_id is None, get all of the images

    raises BadImageFolder
    """
    select_all_images = select(Image).options(joinedload(Image.raw_file))
    images: Sequence[Image]
    if image_ids is None:
        images = session.execute(select_all_images).scalars().all()

    else:
        images = []
        for image_id in image_ids:
            stmt = select_all_images.where(Image.folder == image_id)
            try:
                images.append(session.execute(stmt).scalar_one())
            except NoResultFound as e:
                raise BadImageFolder(image_id) from e
    return images


def get_path(session: Session, record: Channel | Image | None) -> Path:
    """
    returns the folder path or file path of (resepectivly) a channel or image

    Raises no prefix_dir
    """
    existing_record = session.execute(
        select(GlobalConfig).filter(GlobalConfig.key == "prefix_dir")
    ).scalar_one_or_none()
    if existing_record is None:
        raise UninitializedDatabase()
    prefix_dir = Path(existing_record.value)
    if record is None:
        return prefix_dir
    suffix = (
        Path(record.image.folder) / record.path
        if isinstance(record, Channel)
        else record.folder
    )
    return prefix_dir / suffix


def check_progress(
    session: Session,
    input_channels: Sequence[Channel],
    current_step: int,
) -> Literal["redoing", "ok"]:
    """
    checks if the progress of the image is compatible with current step

    Raises
        BadInputImages
        SkippingStep
    """
    session.add_all(input_channels)
    image = input_channels[0].image
    if max(i.step_number for i in input_channels) < current_step - 1:
        raise BadInputImages([get_path(session, i) for i in input_channels])
    if image.progress != current_step - 1:
        if image.progress >= current_step:
            return "redoing"
        else:
            raise SkippingStep(current_step, image.progress)
    return "ok"


def perform_analysis_step(
    session: Session,
    analysis_step: AnalysisStep,
    input_channels: list[Channel],
    output_channels: list[Channel],
    current_step: int,
    copy_scale=False,
) -> bool:
    """
    Wires all of objects together mutating the image. All relationships are
    filled out by this function, and image progress is mutated. Adds all of the
    objects to session
    copy_scale determines whether to copy the scale from input image

    This function also fills out step_number, image, number on output_channels, and adds id to path

    if the image is not in the right progress, it returns false else it returns True
    """
    # process output_channels
    if not all(o.producer is None for o in output_channels):
        raise ValueError("Channels have already been produced")
    for oc in output_channels:
        if oc.number is None:
            oc.number = input_channels[0].number
        oc.step_number = current_step
    if any(not get_path(session, channel).exists() for channel in input_channels):
        raise FileNotFoundError(f"missing {input_channels}")
    if copy_scale:
        for oc in output_channels:
            oc.scalex = input_channels[0].scalex
            oc.scaley = input_channels[0].scaley
            oc.scalez = input_channels[0].scalez
    # make connections
    image = input_channels[0].image
    for output_channel in output_channels:
        output_channel.image = image
    session.add_all(input_channels + output_channels + [analysis_step])
    session.flush()
    # add id to path
    for oc in output_channels:
        assert oc.id is not None
        oc.path = f"{oc.id:04}{oc.path}"
    # see if input images are appropreate
    progress_status = check_progress(session, input_channels, current_step)
    if progress_status in ["bad inputs", "skipping"]:
        return False
    image.progress = current_step
    analysis_step.output_channels = output_channels
    analysis_step.input_channels = input_channels
    logger.debug("Completed step %d on %s", current_step, image.folder)
    return True


def validate_db(session: Session):
    """
    checks to see if a database has tables or is empty
    if its empty, raises UninitializedDatabase
    """
    tables = inspect(session.get_bind()).get_table_names()
    if len(tables) == 0:
        raise UninitializedDatabase("You must run `t6alh init` first")


def save_channel_to_disk(session: Session, channel: Channel, data: np.ndarray):
    """
    saves a channel to disk as a nrrd
    """
    header = {
        "space": "RAS",
        "sample units": ("micron", "micron", "micron"),
        "space directions": np.diag([channel.scalez, channel.scaley, channel.scalex]),
        "labels": ["Z", "Y", "X"],
    }
    session.flush()
    assert channel.id is not None
    channel_path = get_path(session, channel)
    logger.info(f"writing {channel_path}")
    nrrd.write(
        str(channel_path),
        data=data,
        header=header,
        compression_level=2,
    )


def get_mask_template_path(session: Session) -> Path:
    """
    gets the path to the mask_template
    """
    existing_record = session.execute(
        select(GlobalConfig).filter(GlobalConfig.key == "mask_template_path")
    ).scalar_one_or_none()
    if existing_record is not None:
        return Path(existing_record.value)
    existing_record = session.execute(
        select(GlobalConfig).filter(GlobalConfig.key == "prefix_dir")
    ).scalar_one_or_none()
    if existing_record is None:
        raise UninitializedDatabase()
    template_path = Path(existing_record.value) / "template/mask_template.nrrd"
    if not template_path.exists():
        raise CannotFindTemplate()
    return template_path
