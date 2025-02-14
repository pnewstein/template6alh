"""
Tools to do iterative mask averaging
"""

from typing import Literal, Sequence
from pathlib import Path
import shutil

import nrrd

from . import utils

ZERO_INIT = """! TYPEDSTREAM 2.4

affine_xform {
    xlate 0 0 0
    rotate 0 0 0
    scale 1 1 1
    shear 0 0 0
    center 0 0 0
}"""

def get_grid_spacing(md: dict, iteration: int) -> float:
    """
    gets half the smallest grid spacing as done in get_level_warp_args from cmtk iterave_shape_averaging
    """
    spacings = utils.get_spacings(md)
    sizes = md["sizes"]
    fov = spacings * sizes
    return 5 * fov.min() / (2**iteration)


def do_iteration(
    input_images: Sequence[Path],
    prev_dir: Path | None,
    new_dir: Path,
    mode: Literal["warp", "affine", "none"],
    iteration: int,
) -> Path:
    """
    does an iteration puting all warps and intermediate images and new templates
    returns the path to the new template
    """
    new_dir.mkdir(exist_ok=True)
    new_template = new_dir / "average.nrrd"
    if mode == "none":
        out_images = list(input_images)
        xforms = [(new_dir / i.name).with_suffix(".xform") for i in input_images]
        for xform in xforms:
            xform.write_text(ZERO_INIT)
    elif mode == "affine":
        assert prev_dir is not None
        prev_template = prev_dir / "average.nrrd"
        assert prev_template.exists()
        xforms = [(new_dir / i).with_suffix(".xform") for i in input_images]
        out_images = [new_dir / i for i in input_images]
        for input_path, xform_path, output_path in zip(
            input_images, xforms, out_images
        ):
            utils.run_with_logging(
                (
                    utils.get_cmtk_executable("registration"),
                    "--initxlate",
                    "--dofs",
                    "6,9",
                    "--auto-multi-levels",
                    "5",
                    "-o",
                    xform_path,
                    prev_template,
                    input_path,
                )
            )
            utils.run_with_logging(
                (
                    utils.get_cmtk_executable("reformatx"),
                    "-o",
                    output_path,
                    "--floating",
                    input_path,
                    prev_template,
                    xform_path,
                )
            )
    elif mode == "warp":
        assert prev_dir is not None
        prev_template = prev_dir / "average.nrrd"
        assert prev_template.exists()
        xforms = [(new_dir / i.name).with_suffix(".xform") for i in input_images]
        out_images = [new_dir / i.name for i in input_images]
        prev_xforms = [(prev_dir / i.name).with_suffix(".xform") for i in input_images]
        grid_spacing = get_grid_spacing(nrrd.read_header(str(prev_template)), iteration)
        for input_path, xform_path, output_path, prev_xform_path in zip(
            input_images, xforms, out_images, prev_xforms
        ):
            if xform_path.exists():
                continue
            utils.run_with_logging(
                (
                    utils.get_cmtk_executable("warp"),
                    "--threads",
                    "64",
                    "--echo",
                    "-v",
                    "--refine",
                    "1",
                    "--delay-refine",
                    "--energy-weight",
                    "1e-1",
                    "--grid-spacing",
                    str(grid_spacing),
                    "--initial",
                    prev_xform_path,
                    "-o",
                    xform_path,
                    prev_template,
                    input_path,
                )
            )
            utils.run_with_logging(
                (
                    utils.get_cmtk_executable("reformatx"),
                    "-o",
                    output_path,
                    "--floating",
                    input_path,
                    prev_template,
                    xform_path,
                )
            )
    if mode == "none":
        shutil.copy(input_images[0], new_template)
    else:
        if not new_template.exists():
            utils.run_with_logging(
                [utils.get_cmtk_executable("average_images"), "-o", new_template] + out_images
            )
    return new_dir
