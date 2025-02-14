"""
API functions specificaly for template creations
"""

from datetime import datetime
import re
from logging import getLogger
from pathlib import Path

from sqlalchemy.orm import Session, aliased
from sqlalchemy import select
import numpy as np
import nrrd
import click

from .sql_classes import Channel, Image, AnalysisStep, ChannelMetadata
from .execptions import InvalidStepError
from .sql_utils import (
    perform_analysis_step,
    get_path,
    validate_db,
    get_imgs,
    get_mask_template_path,
    check_progress,
    ConfigDict,
    select_most_recent,
    select_recent_landmark_xform_and_mask,
)
from .utils import get_cmtk_executable, run_with_logging, get_target_grid
from .iterative_shape_averaging import do_iteration
from . import matplotlib_slice, api

logger = getLogger(__name__)


def landmark_align(
    session: Session,
    image_paths: list[str] | None,
    landmark_path: Path | None,
    target_grid: str | None,
):
    """
    does registration (uing api.landmark_register) and reformating
    """
    config_dict = ConfigDict(session)
    if landmark_path is None:
        landmark_path = (
            Path(config_dict["prefix_dir"]) / "template/mask_template.landmarks"
        )
        landmark_path.parent.mkdir(parents=True, exist_ok=True)
        # prevent error when getting path later on
        landmark_path.with_suffix(".nrrd").touch()
        landmark_path.write_text(
            "20.0 20.0 50.0 brain\n20.0 20.0 50.0 sez\n30.0 170.0 50.0 tip"
        )
    config_dict["mask_template_landmarks_path"] = str(landmark_path.resolve())
    if target_grid is None:
        target_grid = "90,450,250:0.5000,0.5000,0.5000"
    (match,) = re.finditer(r".*:(.+),(.+),(.+)", target_grid)
    z_str, y_str, x_str = match.groups()
    api.landmark_register(session, image_paths)
    images = get_imgs(session, image_paths)
    for image in images:
        xform_mask_chan = session.execute(
            select_recent_landmark_xform_and_mask(image)
        ).first()
        if xform_mask_chan is None:
            logger.warning("could not find a best flip for image %s", image.folder)
            continue
        xform, mask_chan = xform_mask_chan
        assert mask_chan.channel_type == "mask"
        assert xform.channel_type == "xform"
        output_channel = Channel()
        output_channel.path = "landmark_reformat.nrrd"
        output_channel.channel_type = "aligned-mask"
        output_channel.scalez = float(z_str)
        output_channel.scaley = float(y_str)
        output_channel.scalex = float(x_str)
        step = AnalysisStep()
        step.function = "landmark-align"
        step.kwargs = repr({"landmark_path": landmark_path, "target_grid": target_grid})
        step.runtime = datetime.now()
        perform_analysis_step(
            session,
            step,
            [xform, mask_chan],
            [output_channel],
            3,
            update_progress=False,
        )
        run_with_logging(
            (
                get_cmtk_executable("reformatx"),
                "-o",
                get_path(session, output_channel),
                "--target-grid",
                target_grid,
                "--floating",
                get_path(session, mask_chan),
                get_path(session, xform),
            )
        )
    session.commit()


def iterative_mask_template(
    session: Session, image_paths: list[str] | None, make_template=True
):
    """
    takes previously selected templates and makes a groupwise template using cmtk
    """
    if image_paths is None:
        raise InvalidStepError("You must select images to be made into the template")
    validate_db(session)
    images = get_imgs(session, image_paths)
    channels: list[Channel] = []
    for image in images:
        reformat_chan = (
            session.execute(select_most_recent("landmark-align", image))
            .scalars()
            .first()
        )
        if reformat_chan is None:
            logger.warning("could not find a best flip for image %s", image.folder)
            continue
        channels.append(reformat_chan)
    assert len(channels) != 0
    # make calls to cmtk
    sh_args = [get_cmtk_executable("iterative_shape_averaging")] + [
        get_path(session, c) for c in channels
    ]
    cwd = get_path(session, None)
    # add path to config_dict
    config_dict = ConfigDict(session)
    if "mask_template_path" not in config_dict:
        config_dict["mask_template_path"] = str(
            Path(config_dict["mask_template_landmarks_path"]).with_suffix(".nrrd")
        )
    session.commit()
    mask_path = Path(config_dict["mask_template_path"])
    if make_template:
        run_with_logging(sh_args, cwd=cwd)
        paths = list((cwd / "isa/pass5").glob("*landmark_reformat.nii.gz"))
        assert len(paths) != 0
        sh_args = [
            get_cmtk_executable("average_images"),
            "--outfile-name",
            mask_path,
        ] + paths
        run_with_logging(sh_args, cwd=cwd)
    # postprocesses template
    assert mask_path.exists()
    assert mask_path.stat().st_size > 1
    template, template_md = nrrd.read(str(mask_path))
    template = template * (254 / template.max())
    template = template.astype(np.uint8)
    nrrd.write(file=str(mask_path), data=template, header=template_md)


def fasii_groupwise(session: Session, image_paths: list[str] | None):
    """
    does a groupwise warp to create a fasII tempalate
    """
    images = get_imgs(session, image_paths)
    warpeds: list[Channel] = []
    for image in images:
        warped = (
            session.execute(
                select_most_recent("mask-register", image).filter(
                    Channel.channel_type == "aligned",
                )
            )
            .scalars()
            .first()
        )
        if warped is None:
            logger.warning("No results for image %s", image.folder)
            continue
        warpeds.append(warped)
    if len(warpeds) == 0:
        raise InvalidStepError("No images to process")
    # preprocess image
    prefix_dir = get_path(session, None)
    run_with_logging(
        [
            get_cmtk_executable("groupwise_init"),
            "-O",
            prefix_dir / "groupwise/initial",
            "-v",
        ]
        + [get_path(session, w) for w in warpeds]
    )
    run_with_logging(
        (
            "gunzip",
            "-f",
            prefix_dir / "groupwise/initial/groupwise.xforms.gz",
        )
    )
    run_with_logging(
        (
            get_cmtk_executable("groupwise_warp"),
            "--congeal",
            "-O",
            prefix_dir / "groupwise/warp",
            "-v",
            "--match-histograms",
            "--histogram-bins",
            "32",
            "--grid-spacing",
            "40",
            "--grid-spacing-fit",
            "--refine-grid",
            "5",
            "--zero-sum-no-affine",
            "--downsample-from",
            "8",
            "--downsample-to",
            "1",
            "--exploration",
            "6.4",
            "--accuracy",
            "0.1",
            "--force-background",
            "0",
            "--output-average",
            "template.nrrd",
            prefix_dir / "groupwise/initial/groupwise.xforms",
        )
    )


def fasii_template(session: Session, image_paths: list[str] | None):
    """
    does a groupwise warp to create a fasII tempalate
    """
    api.mask_register(session, image_paths)
    images = get_imgs(session, image_paths)
    warpeds: list[Channel] = []
    for image in images:
        warped = (
            session.execute(
                select_most_recent("mask-register", image).filter(
                    Channel.channel_type == "aligned",
                )
            )
            .scalars()
            .first()
        )
        if warped is None:
            logger.warning("No results for image %s", image.folder)
            continue
        warpeds.append(warped)
    if len(warpeds) == 0:
        raise InvalidStepError("No images to process")
    # preprocess image
    prefix_dir = get_path(session, None)
    template_create_dir = prefix_dir / "template_create"
    template_create_dir.mkdir(exist_ok=True)
    input_images = [get_path(session, w) for w in warpeds]
    prev_dir = do_iteration(
        input_images,
        None,
        template_create_dir / "iter0",
        mode="none",
        iteration=0,
    )
    for i in range(1, 6):
        prev_dir = do_iteration(
            input_images, prev_dir, template_create_dir / f"iter{i}", "warp", i
        )



def write_landmarks(session: Session):
    """
    opens a gui to write landmarks for the template path to disk
    """
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


def advance_images(session: Session, image_paths: list[str] | None):
    """
    for testing purposes, advance the image progress
    """
    for image in get_imgs(session, image_paths):
        image.progress = image.progress + 1
    session.commit()
