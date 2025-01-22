"""
tests sql
"""

import tempfile
from pathlib import Path
from datetime import datetime
import time

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, select
import pytest

from template6alh.logger import logger
from template6alh.sql_classes import (
    Base,
    Image,
    Channel,
    GlobalConfig,
    RawFile,
    AnalysisStep,
)
from template6alh.sql_utils import (
    perform_analysis_step,
    LogEntry,
    get_log,
    get_path,
    get_mask_template_path,
)
from template6alh.execptions import BadInputImages, CannotFindTemplate


def get_session(folder: str) -> Session:
    """
    gets a session for testing
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    global_config = GlobalConfig(key="prefix_dir", value=folder)
    session = Session(engine)
    if session.query(GlobalConfig).count() == 0:
        session.add(global_config)
    return session


def get_image(session: Session, folder: str) -> Image:
    """
    makes an example image in the session with a new raw data
    """
    channel = Channel(
        path="1.nrrd",
        number=1,
        channel_type="raw",
        step_number=0,
        scalex=1,
        scaley=1,
        scalez=1,
    )
    image = Image(
        progress=0,
        folder=folder,
        channels=[channel],
    )
    raw_data = RawFile(path="raw.czi", fasii_chan=1, images=[image])
    session.add(raw_data)
    session.add(image)
    session.commit()
    get_path(session, image).mkdir(exist_ok=True)
    get_path(session, channel).touch()
    return image


def make_compelted_analysis_step(
    session: Session, image: Image, step_number=1
) -> AnalysisStep:
    """
    makes a completed AnalysisStep
    """
    image_progress = image.progress
    input_channels = [
        Channel(
            path=f"in{step_number}",
            channel_type="raw",
            step_number=step_number - 1,
            number=1,
            scalex=1,
            scaley=1,
            scalez=1,
            image=image,
        )
    ]
    output_channels = [
        Channel(
            path=f"out{step_number}",
            channel_type="raw",
            step_number=step_number,
            scalex=1,
            scaley=1,
            scalez=1,
            image=image,
        )
    ]
    step = AnalysisStep(
        function=f"step{step_number}", kwargs="{}", runtime=datetime.now()
    )
    session.add_all(input_channels + output_channels)
    for channel in input_channels + output_channels:
        get_path(session, channel).touch()
    assert perform_analysis_step(
        session, step, input_channels, output_channels, current_step=step_number
    )
    assert image.progress == image_progress + 1
    return step


def test_making_image():
    with tempfile.TemporaryDirectory() as folder:
        session = get_session(folder)
        image = get_image(session, folder)
        assert image.channels[0].number == 1
        stmt = select(Channel)
        channel = session.execute(stmt).scalar_one()
        assert channel.number == 1
        channel.number = 2
        assert image.channels[0].number == 2


def test_perform_analysis_step():
    with tempfile.TemporaryDirectory() as folder:
        session = get_session(folder)
        image = get_image(session, folder)
        input_channels = [
            Channel(
                path="_",
                channel_type="raw",
                step_number=0,
                number=1,
                scalex=1,
                scaley=1,
                scalez=1,
                image=image,
            )
        ]
        output_channels = [
            Channel(
                path="b",
                channel_type="raw",
                number=1,
                step_number=1,
                scalex=1,
                scaley=1,
                scalez=1,
                image=image,
            )
        ]
        input_channels2 = [
            Channel(
                path="_",
                channel_type="raw",
                step_number=1,
                number=1,
                scalex=1,
                scaley=1,
                scalez=1,
                image=image,
            )
        ]
        output_channels2 = [
            Channel(
                path="c",
                channel_type="raw",
                step_number=2,
                number=1,
                scalex=1,
                scaley=1,
                scalez=1,
                image=image,
            )
        ]
        step = AnalysisStep(function="test", kwargs="{}", runtime=datetime.now())
        step2 = AnalysisStep(function="test2", kwargs="{}", runtime=datetime.now())

        all_channels = (
            input_channels + output_channels + input_channels2 + output_channels2
        )
        session.add_all(all_channels)
        for channel in all_channels:
            get_path(session, channel).touch()
        assert perform_analysis_step(session, step, input_channels, output_channels, 1)
        assert input_channels[0].id == 2
        assert output_channels[0].image.id == image.id
        with pytest.raises(BadInputImages):
            perform_analysis_step(session, step, input_channels, output_channels2, 5)
        assert input_channels[0].used_in_steps[0].function == "test"
        assert output_channels[0].producer.function == "test"
        with pytest.raises(ValueError):
            perform_analysis_step(session, step2, input_channels, output_channels, 2)
        assert perform_analysis_step(
            session, step2, input_channels + input_channels2, output_channels2, 2
        )
        assert input_channels[0].used_in_steps[1].function == "test2"


def test_partial_channel():
    chan = Channel()
    assert chan.number is None


def test_custom_logger():
    logger.warning("UhOH")


def test_logging_step():
    with tempfile.TemporaryDirectory() as folder:
        session = get_session(folder)
        analysis_step = make_compelted_analysis_step(
            session, get_image(session, folder), 1
        )
    assert analysis_step.output_channels[0].step_number == 1
    record = LogEntry.from_step(session, analysis_step)
    assert analysis_step.input_channels[0].step_number == 0
    assert record.step_number == 1
    assert record.input_paths[0].name == analysis_step.input_channels[0].path
    assert record.output_paths[0].name == analysis_step.output_channels[0].path


def test_get_log():
    with tempfile.TemporaryDirectory() as folder:
        session = get_session(folder)
        analysis_step1 = make_compelted_analysis_step(
            session, get_image(session, folder), 1
        )
        image = analysis_step1.input_channels[0].image
        analysis_step2 = AnalysisStep(
            function="f1", kwargs="{}", runtime=datetime.now()
        )
        image.progress = 2
        analysis_step3 = make_compelted_analysis_step(session, image, 3)
        analysis_step4 = AnalysisStep(
            function="f4", kwargs="{}", runtime=datetime.now()
        )
        image.progress = 4
        analysis_step5 = make_compelted_analysis_step(session, image, 5)
        analysis_step2.input_channels = analysis_step1.output_channels
        analysis_step2.output_channels = analysis_step3.input_channels
        analysis_step4.input_channels = (
            analysis_step3.output_channels + analysis_step1.input_channels
        )
        analysis_step4.output_channels = analysis_step5.input_channels
        assert analysis_step1.output_channels[0].step_number == 1
        assert analysis_step2.output_channels[0].step_number == 2
        assert analysis_step3.output_channels[0].step_number == 3
        assert analysis_step4.output_channels[0].step_number == 4
        assert analysis_step5.output_channels[0].step_number == 5
        logs = get_log(session, analysis_step5.output_channels[0])
        assert len(logs) == 5
        assert logs[0].step_number == 1
        assert logs[4].step_number == 5
        assert len(logs[3].input_paths) == 2
        for log in logs:
            print(log.format())


def test_get_mask_template_path():
    with tempfile.TemporaryDirectory() as folder:
        session = get_session(folder)
        with pytest.raises(CannotFindTemplate):
            get_mask_template_path(session)
        mt_path = Path(folder) / "template/mask_template.nrrd"
        mt_path.parent.mkdir(exist_ok=True)
        mt_path.touch(exist_ok=True)
        assert get_mask_template_path(session) == mt_path


if __name__ == "__main__":
    test_get_log()
