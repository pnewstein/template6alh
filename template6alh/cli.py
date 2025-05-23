"""
defines the command line interface
"""

from pathlib import Path
import json
import sys
import nrrd

import click
from sqlalchemy.orm import Session

from . import api, template_creation, matplotlib_slice, sql_utils
from .logger import logger
from .execptions import InvalidStepError
from .utils import get_engine_with_context, image_folders_from_file, get_spacings


@click.group()
@click.option(
    "-v", "--verbose", is_flag=True, help="Enable verbose mode.", default=False
)
@click.option(
    "-d",
    "--database",
    type=click.Path(dir_okay=False),
    help="Specify a path to a database to use",
    default=None,
)
@click.pass_context
def main(ctx: click.Context, verbose: bool, database: str | None):
    ctx_dict = ctx.ensure_object(dict)
    ctx_dict["database"] = None if database is None else Path(database)
    if verbose:
        assert logger.parent is not None
        handler = next(h for h in logger.parent.handlers if h.name == "stderr")
        handler.setLevel("DEBUG")
    logger.info("New cli innovation")


@main.command()
@click.pass_context
def get_paths(ctx: click.Context):
    """
    prints relevant paths as a json file

    \b
    t6alh get-paths
    """

    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    with Session(get_engine_with_context(ctx_dict)) as session:
        paths = api.get_paths(session, ctx_dict.get("database"))
    click.echo(json.dumps({k: str(v) for k, v in paths.items()}, indent=4))


@main.command()
@click.argument("raw-files", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option(
    "-r",
    "--root-dir",
    type=click.Path(file_okay=False),
    help="the path where data is cached",
)
@click.option(
    "-n", "--neuropil-chan", type=int, help="Channel labeling the neuropil (1 indexed)"
)
@click.option(
    "-f",
    "--fasii-chan",
    type=int,
    default=None,
    help="Channel labeling the FasII tracts (1 indexed)",
)
@click.option(
    "-e",
    "--eve-chan",
    type=int,
    default=None,
    help="Channel labeling even skipped (1 indexed)",
)
@click.option(
    "-s",
    "--new-scale",
    type=float,
    default=None,
    help="the output scale in um. Default is 1. Negative numbers are interpreted as no new scale",
)
@click.option(
    "-o",
    "--opening-size",
    type=float,
    default=None,
    help="the size for which smaller objects will be removed. Default 1um",
)
@click.option(
    "-g",
    "--gamma",
    type=float,
    default=None,
    help="gamma correction for the image. default: 2",
)
@click.pass_context
def init_and_segment(
    ctx: click.Context,
    raw_files: tuple[str, ...],
    root_dir: str,
    neuropil_chan: int,
    fasii_chan: int | None,
    eve_chan: int | None,
    new_scale: float | None,
    gamma: float | None,
    opening_size: float | None,
):
    """
    Takes a list of microscopy files and initializes the db making a cache of
    all of the images, also segments the image.

    \b
    t6alh init-and-segment /storage/1.czi /storage/2.czi --root-dir ~/t6alh-cache --neuropil-chan 1 --fasii-chan 2
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    if not raw_files:
        click.echo("No raw files", err=True)
    try:
        engine = get_engine_with_context(ctx_dict, delete_previous=True)
        api.init(
            engine,
            raw_data=[Path(f) for f in raw_files],
            root_dir=Path(root_dir),
            fasii_chan=fasii_chan,
            neuropil_chan=neuropil_chan,
            eve_chan=eve_chan,
        )
        with Session(engine) as session:
            api.segment_neuropil(session, None, new_scale, gamma, opening_size)
    except InvalidStepError as e:
        click.echo(e)
        sys.exit(1)


@main.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.option(
    "-c",
    "--channel",
    type=click.INT,
    multiple=True,
    help="Channel can be allowed. Multiple are supported '-c 1 -c 2'",
)
@click.pass_context
def align(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
    channel: tuple[int, ...],
):
    """
    Does a multi step alignment

    \b
    t6alh align 001 002 -c 1 -c 2 -c 3 -c 4
    """

    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.landmark_register(session, image_folders_or_none)
            api.mask_register(session, image_folders_or_none)
            api.select_neuropil_fasii(
                session,
                image_paths=image_folders_or_none,
            )
            api.fasii_align(session, image_folders_or_none, channel)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command()
@click.argument("raw-files", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option(
    "-r",
    "--root-dir",
    type=click.Path(file_okay=False),
    help="the path where data is cached",
)
@click.option(
    "-n", "--neuropil-chan", type=int, help="Channel labeling the neuropil (1 indexed)"
)
@click.option(
    "-f",
    "--fasii-chan",
    type=int,
    default=None,
    help="Channel labeling the FasII tracts (1 indexed)",
)
@click.option(
    "-e",
    "--eve-chan",
    type=int,
    default=None,
    help="Channel labeling even skipped (1 indexed)",
)
@click.pass_context
def init(
    ctx: click.Context,
    raw_files: tuple[str, ...],
    root_dir: str,
    neuropil_chan: int,
    fasii_chan: int | None,
    eve_chan: int | None,
):
    """

    Takes a list of microscopy files and initializes a sqlite db making a cache
    of all channels of all images in files like `001/chan1`

    \b
    t6alh init /storage/1.czi /storage/2.czi --root-dir ~/t6alh-cache --neuropil-chan 1 --fasii-chan 2
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    if not raw_files:
        click.echo("No raw files", err=True)
    try:
        api.init(
            get_engine_with_context(ctx_dict, delete_previous=True),
            raw_data=[Path(f) for f in raw_files],
            root_dir=Path(root_dir),
            fasii_chan=fasii_chan,
            neuropil_chan=neuropil_chan,
            eve_chan=eve_chan,
        )
    except InvalidStepError as e:
        click.echo(e)
        sys.exit(1)


@main.command()
@click.argument("raw-files", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option(
    "-n", "--neuropil-chan", type=int, help="Channel labeling the neuropil (1 indexed)"
)
@click.option(
    "-f",
    "--fasii-chan",
    type=int,
    default=None,
    help="Channel labeling the FasII tracts (1 indexed)",
)
@click.option(
    "-e",
    "--eve-chan",
    type=int,
    default=None,
    help="Channel labeling even skipped (1 indexed)",
)
@click.pass_context
def add_more_raw(
    ctx: click.Context,
    raw_files: tuple[str, ...],
    neuropil_chan: int,
    fasii_chan: int | None,
    eve_chan: int | None,
):
    """
    Takes a list of microscopy files to add to the db. Uses similar arguments
    to t6alh init

    \b
    t6alh init /storage/1.czi /storage/2.czi --neuropil-chan 1 --fasii-chan 2
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    if not raw_files:
        click.echo("No raw files", err=True)
    with Session(get_engine_with_context(ctx_dict)) as session:
        if not raw_files:
            click.echo("No raw files", err=True)
        try:
            api.add_more_raw(
                session,
                raw_data=[Path(f) for f in raw_files],
                fasii_chan=fasii_chan,
                neuropil_chan=neuropil_chan,
                eve_chan=eve_chan,
            )
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.option(
    "-s",
    "--new-scale",
    type=float,
    default=None,
    help="the output scale in um. Default is 1. Negative numbers are interpreted as no new scale",
)
@click.option(
    "-o",
    "--opening-size",
    type=float,
    default=None,
    help="the size for which smaller objects will be removed. Default 2um",
)
@click.option(
    "-g",
    "--gamma",
    type=float,
    default=None,
    help="gamma correction for the image. Default: 1",
)
@click.pass_context
def segment_neuropil(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
    new_scale: float | None,
    gamma: float | None,
    opening_size: float | None,
):
    """
    Segments the neuropil labeling each pixel as 0 for not neuropil and 1 for
    neuropil on a scaled down copy of the image creates the file
    `neuropil-mask.nrrd`

    \b
    t6alh segment_neuropil 001 002 --opening-size 2
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.segment_neuropil(
                session,
                image_paths=image_folders_or_none,
                new_scale=new_scale,
                gamma=gamma,
                opening_size=opening_size,
            )
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command()
@click.pass_context
def clean(ctx: click.Context):
    """
    removes all files from database which are no longer on the disk

    \b
    t6alh clean
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.clean(session)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.pass_context
def make_landmarks(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    """
    Uses a image slicer to find landmarks on all segmented images. Creates a
    `.landmarks` file. Requires `$DISPLAY` to be set

    \b
    t6alh make_landmarks 001 002
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.make_landmarks(session, image_folders_or_none)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.pass_context
def landmark_register(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    """
    uses `fit_affine_xform_landmarks` to register to the template, creates a
    `landmark.xform` file describing the transformation

    \b
    t6alh landmark_register 001 002
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.landmark_register(session, image_folders_or_none)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.pass_context
def mask_register(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    """
    Uses `regiser` and `warp` to fit the registration from each image to the
    mask template, creating `affine_mask.xform` and `warp_mask.xform`, then
    uses `reformatx` to reformat the fasII tracts creating
    `mask_warped_fasii.nrrd`

    \b
    t6alh mask-register 001 002
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.mask_register(session, image_folders_or_none)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.option(
    "-c",
    "--channel",
    type=click.INT,
    multiple=True,
    help="Channel can be allowed. Multiple are supported '-c 1 -c 2'",
)
@click.pass_context
def fasii_align(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
    channel: tuple[int, ...],
):
    """
    Aligns the image to the FasII template using `warp` between the template
    and the mask-aligned image creating 'warp.xform', then registers each of
    the specified channels to the complete warp making files following the
    pattern `reformated_chanN.nrrd`

    \b
    t6alh fasii-align 001 001 -c 1 -c 2 -c 3
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.fasii_align(session, image_folders_or_none, channel)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command()
@click.argument("images", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option(
    "-g", "gamma", type=float, default=None, help="Gamma correction. Default=1"
)
@click.option(
    "-s",
    "scale",
    type=float,
    default=None,
    help="scale ratio. 0.5 means half as many pixels in each axis",
)
def view(images: list[str], scale: float | None, gamma: float | None):
    """
    Displays the image (or a low resolution version of the image) on a slicer.
    $DISPLAY must be set

    \b
    t6alh view -s 0.1 image.nrrd
    """
    slicers = []
    for image in images:
        image_path = Path(image)
        if image_path.suffix not in (".nrrd", ".nhdr"):
            click.echo(f"{image_path} is not an nrrd", err=True)
            continue
        data, md = nrrd.read(image)
        if scale is not None:
            scale_frac = tuple((get_spacings(md) / scale).tolist())
        else:
            scale_frac = scale
        slicers.append(
            matplotlib_slice.get_slicer(data, image, gamma=gamma, scale=scale_frac)
        )
    click.confirm("Close all windows?")
    for slicer in slicers:
        slicer.quit()


@main.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.pass_context
def select_neuropil_fasii(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    """
    Creates an image where all of the pixels in the FasII channel outside of
    `neuropil_mask.nrrd` are set to 0. Saves this image as
    `neuropil_fasii.nrrd`

    \b
    t6alh select-neuropil-fasii 001 002
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.select_neuropil_fasii(
                session,
                image_paths=image_folders_or_none,
            )
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command()
@click.pass_context
@click.option("-j", "--json", is_flag=True, help="output as json")
@click.option("-t", "--text", is_flag=True, help="output as raw text. Default")
def raw_data_info(ctx: click.Context, json: bool, text: bool):
    """
    prints information about the raw data files. Specificaly it prints the
    fasii channel number, the neuropil channel number, and a list of all of the
    image paths

    \b
    t6alh -d test.db raw-data-info --json
    """
    if json and text:
        raise click.UsageError("Options json and text are mutually exclusive")
    output = "json" if json else "text"
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            click.echo(api.raw_data_info(session, output))
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.group(help="Commands for creating a template image")
@click.pass_context
def template(ctx: click.Context):
    _ = ctx
    pass


@template.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.option(
    "-l",
    "--landmark-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Landmarks for the template image. Reasonable default to center a Drosophila VNC",
)
@click.option(
    "-t",
    "--target-grid",
    type=str,
    default=None,
    help="target-grid parameter passed to reformatx."
    "Reasonable default to center a Drosophila VNC: 80,451,200:0.5000,0.5000,0.5000",
)
@click.pass_context
def landmark_align(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
    landmark_path: str | None,
    target_grid: str | None,
):
    """
    Takes landmarks from `t6alh make-landmarks` to align each image using
    `fit_affine_xform_landmarks` then aligns the mask using `reformatx` to get
    `landmark_reformat.nrrd`

    \b
    t6alh template landmark-align 001 002 --target-grid 80,451,200:0.5,0.50,0.5
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            sql_utils.validate_db(session)
            template_creation.landmark_align(
                session,
                image_paths=image_folders_or_none,
                landmark_path=None if landmark_path is None else Path(landmark_path),
                target_grid=target_grid,
            )
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@template.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.pass_context
def make_mask_template(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    """
    Uses `iterative_shape_averaging` to align and warp together
    `landmark_reformat.nrrd` to create `template/mask_template.nrrd`

    \b
    t6alh template make-mask-template 001 002 003 004
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            template_creation.iterative_mask_template(session, image_folders_or_none)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@template.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.pass_context
def fasii_template(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    """
    Uses a iterative process to warp and `mask_warped_fasii.nrrd` images
    created by `t6alh mask-register` into a final template. Creates the file
    `template/fasii_template.nrrd`. Caches intermediate results, since this
    command takes a long time to run


    \b
    t6alh template fasii-template 001 002 005
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            template_creation.fasii_template(
                session,
                image_paths=image_folders_or_none,
            )
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@template.command()
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternative to [IMAGE-FOLDERS]",
)
@click.pass_context
def advance_images(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    """
    advance the image for testing purposes
    """
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            template_creation.advance_images(
                session,
                image_paths=image_folders_or_none,
            )
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)
