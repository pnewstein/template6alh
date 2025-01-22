"""
defines the command line interface
"""

from pathlib import Path
import json
import sys
from typing import get_args

import click
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound

from . import api, template_creation
from .logger import logger
from .execptions import InvalidStepError, UninitializedDatabase
from .utils import get_engine_with_context, image_folders_from_file


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
    ctx_dict["database"] = database
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
    "--filter-sigma",
    type=float,
    default=None,
    help="a sigma for a noise removal filter: default .1 um",
)
@click.pass_context
def segment_neuropil(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
    new_scale: float | None,
    filter_sigma: float | None,
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
                filter_sigma=filter_sigma,
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
    Alignes mask to template mask with affine xform and selects the best flip
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
def mask_affine(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            api.mask_affine(session, image_folders_or_none)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


def validate_flip(ctx, param, value):
    _ = ctx, param
    valid_flips = get_args(template_creation.FlipLiteral)
    if value not in valid_flips:
        raise click.BadParameter(f"Should be one of {repr(valid_flips)}")
    return value


@main.group(help="Commands for creating a template image")
@click.pass_context
def template(ctx: click.Context):
    pass


@template.command(
    help="""
Selects images for use in a template.
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
    "--flip",
    type=str,
    callback=validate_flip,
    help="axes to flip, eg '001' would mean flip the x axis, '101' would mean flip the z and the x",
)
@click.pass_context
def select_images(
    ctx: click.Context,
    image_folders: list[str],
    image_folders_file: str | None,
    flip: template_creation.FlipLiteral,
):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    image_folders_or_none = image_folders_from_file(image_folders, image_folders_file)
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            template_creation.select_images(
                session,
                image_paths=image_folders_or_none,
                flip=flip,
            )
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)


@template.command(
    help="""
    makes a template from the seleced images
"""
)
@click.pass_context
def make_mask_template(
    ctx: click.Context,
):
    ctx_dict = ctx.find_object(dict)
    assert ctx_dict is not None
    with Session(get_engine_with_context(ctx_dict)) as session:
        try:
            template_creation.iterative_mask_template(session)
        except InvalidStepError as e:
            click.echo(e)
            sys.exit(1)
