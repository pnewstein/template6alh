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


@main.command(help="prints relevent paths as json")
@click.pass_context
def get_paths(ctx: click.Context):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    with Session(get_engine_with_context(ctx_dict)) as session:
        paths = api.get_paths(session, ctx_dict.get("database"))
    click.echo(json.dumps({k: str(v) for k, v in paths.items()}, indent=4))

@main.command(
    "init-and-segment",
    help="Takes a list of microscopy files and initializes the db making a cache of all of the images, also segments the image",
)
@click.argument("raw-files", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option(
    "-r",
    "--root-dir",
    type=click.Path(file_okay=False),
    help="the path where data is cached.",
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
    help="Channel labeling the FasII tracts (1 indexed)",
)
@click.option(
    "-s",
    "--new-scale",
    type=float,
    default=None,
    help="the output scale in um. Default is 1. negative numbers are interpreted as no new scale",
)
@click.option(
    "-o",
    "--opening-size",
    type=float,
    default=None,
    help="the size for which smaller objects will be removed. default 1um",
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

@main.command(
    "init",
    help="Takes a list of microscopy files and initializes the db making a cache of all of the images",
)
@click.argument("raw-files", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option(
    "-r",
    "--root-dir",
    type=click.Path(file_okay=False),
    help="the path where data is cached.",
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
    help="Channel labeling the FasII tracts (1 indexed)",
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


@main.command(help="Takes a list of microscopy files to add to the db")
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
    help="Channel labeling the FasII tracts (1 indexed)",
)
@click.pass_context
def add_more_raw(
    ctx: click.Context,
    raw_files: tuple[str, ...],
    neuropil_chan: int,
    fasii_chan: int | None,
    eve_chan: int | None,
):
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


@main.command(
    help="""
Segments the neuropil labeling each pixel as 0 for not neuropil and 1 for neuropil.
Takes a list of folders names.
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.option(
    "-s",
    "--new-scale",
    type=float,
    default=None,
    help="the output scale in um. Default is 1. negative numbers are interpreted as no new scale",
)
@click.option(
    "-o",
    "--opening-size",
    type=float,
    default=None,
    help="the size for which smaller objects will be removed. default 1um",
)
@click.option(
    "-g",
    "--gamma",
    type=float,
    default=None,
    help="gamma correction for the image. default: 2",
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


@main.command(
    help="removes all files from database which are no longer on the disk",
)
@click.pass_context
def clean(ctx: click.Context):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.clean(session)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command(
    help="""
    use a image slicer to find lanmarks on all segmented images
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.pass_context
def make_landmarks(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.make_landmarks(session, image_folders_or_none)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command(
    help="""
    uses landmarks to register to the template
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.pass_context
def landmark_register(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.landmark_register(session, image_folders_or_none)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command(
    help="""
    registers all of the templates masks also reformat fasii image
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.pass_context
def mask_register(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.mask_register(session, image_folders_or_none)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command(
    help="""
    affine transform an image
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.option(
    "-c",
    "--channel",
    type=click.INT,
    multiple=True,
    help="Channel can be allowed. multiple are supported '-c 1 -c 2'",
)
@click.pass_context
def fasii_align(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
    channel: tuple[int, ...],
):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.fasii_align(session, image_folders_or_none, channel)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@main.command(
    help="""
    visualize nrrd files
"""
)
@click.argument("images", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option("-g", "gamma", type=float, default=None)
@click.option("-s", "scale", type=float, default=None)
def view(images: list[str], scale: float | None, gamma: float | None):
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


@main.command(
    help="""
    mask fasII raw with neuropil mask
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.pass_context
def select_neuropil_fasii(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
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


@main.group(help="Commands for creating a template image")
@click.pass_context
def template(ctx: click.Context):
    _ = ctx
    pass


@template.command(
    help="""
    Aligns masks to optional landmark_path using landmarks
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.option(
    "-l",
    "--landmark-path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Landmarks for the template image. Reasonalbe default to center a Drosophila VNC",
)
@click.option(
    "-t",
    "--target-grid",
    type=str,
    default=None,
    help="target-grid parameter passed to reformatx."
    "Reasonalbe default to center a Drosophila VNC: 80,451,200:0.5000,0.5000,0.5000",
)
@click.pass_context
def landmark_align(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
    landmark_path: str | None,
    target_grid: str | None,
):
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


@template.command(
    help="""
    makes a template from the seleced images
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.pass_context
def make_mask_template(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            template_creation.iterative_mask_template(session, image_folders_or_none)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@template.command(
    help="""
    makes a template from mask reformated fasII images
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.pass_context
def fasii_template(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
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


@template.command(
    help="""
    uses a gui to select coordinates of the newly formed template
"""
)
@click.pass_context
def select_template_coordinates(
    ctx: click.Context,
):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            template_creation.write_landmarks(session)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@template.command(
    help="""
    for testing purposes, advance the image
"""
)
@click.argument("image-folders", type=str, nargs=-1)
@click.option(
    "-f",
    "--image-folders-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="A text file with all of the folder names. An alternitive to [IMAGE-FOLDERS]",
)
@click.pass_context
def advance_images(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
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
