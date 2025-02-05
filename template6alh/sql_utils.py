"""
Some utilities for accessing sql
"""

from typing import Sequence, Literal, TypeAlias
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, inspect, Select
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
    InvalidStepError,
)

logger = logging.getLogger(__name__)


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


ConfigKey: TypeAlias = Literal[
    "prefix_dir", "mask_template_path", "mask_template_landmarks_path"
]


class ConfigDict:
    """
    a dict like interface to global config
    """

    def __init__(self, session: Session):
        self.session = session

    def _exising_record_or_raise(self, key: ConfigKey) -> GlobalConfig:
        """
        raises InvalidStepError
        """
        existing_record = self.session.execute(
            select(GlobalConfig).filter(GlobalConfig.key == key)
        ).scalar_one_or_none()
        if existing_record is None:
            raise InvalidStepError(f"Missing key {key} from config")
        return existing_record

    def __getitem__(self, key: ConfigKey) -> str:
        existing_record = self._exising_record_or_raise(key)
        return existing_record.value

    def __setitem__(self, key: ConfigKey, item: str):
        new_record = False
        try:
            existing_record = self._exising_record_or_raise(key)
        except InvalidStepError:
            new_record = True
            existing_record = GlobalConfig()
            existing_record.key = key
        existing_record.value = item
        if new_record:
            self.session.add(existing_record)

    def __contains__(self, key: ConfigKey):
        try:
            self._exising_record_or_raise(key)
            return True
        except InvalidStepError:
            return False


def get_path(session: Session, record: Channel | Image | None) -> Path:
    """
    returns the folder path or file path of (resepectivly) a channel or image

    Raises no prefix_dir
    """
    prefix_dir = Path(ConfigDict(session)["prefix_dir"])
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


def get_mask_template_path(session: Session) -> tuple[Path, Path]:
    """
    gets the path to the mask_template
    returns image_template, landmark_template
    """
    config_dict = ConfigDict(session)
    try:
        return Path(config_dict["mask_template_path"]), Path(
            config_dict["mask_template_landmarks_path"]
        )
    except InvalidStepError:
        pass
    prefix_dir = Path(config_dict["prefix_dir"])
    image_path = prefix_dir / "template/mask_template.nrrd"
    landmarks_path = prefix_dir / "template/mask_template.landmarks"
    if not image_path.exists():
        raise CannotFindTemplate()
    return image_path, landmarks_path


def select_most_recent(
    step_name: str, image: Image, base_select: Select | None = None
) -> Select:
    """
    returns a select statment that joins an input chananel, its producer, and its image sorted by runtime and filtered by image and step name
    """
    if base_select is None:
        base_select = Select(Channel)
    return (
        base_select.join(AnalysisStep, Channel.producer)
        .filter(AnalysisStep.function == step_name)
        .join(Image, Channel.image)
        .filter(Image.folder == image.folder)
        .order_by(AnalysisStep.runtime.desc())
    )
