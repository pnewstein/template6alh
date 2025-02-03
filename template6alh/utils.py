"""
Simple low dependency utilities
"""

import logging
import os
from pathlib import Path
import sys
from typing import Literal, Sequence
from subprocess import run, CompletedProcess

import pandas as pd
from sqlalchemy import Engine, create_engine
import numpy as np
from nrrd.writer import _write_header
import nrrd
import click

from .execptions import ChannelValidationError


logger = logging.getLogger()

FlipLiteral = Literal[
    "000",
    "001",
    "010",
    "011",
    "100",
    "101",
    "110",
    "111",
]


flip_xform_template = """! TYPEDSTREAM 2.4

affine_xform {{
    xlate 0 0 0
    rotate 0 0 0
    scale {z} {y} {x}
    shear 0 0 0
    center 0 0 0
}}"""


def run_with_logging(args: Sequence[str | Path]) -> CompletedProcess[bytes]:
    logger.info(" ".join(str(a) for a in args))
    output = run(args, capture_output=True)
    std_err = output.stderr.decode()
    if std_err:
        logger.info(std_err)
    std_out = output.stdout.decode()
    if std_out:
        logger.info(std_out)
    output.check_returncode()
    return output


def get_flip_xform(flip: FlipLiteral) -> str:
    format = {k: -1 if v == "1" else 1 for v, k in zip(flip, "zyx")}
    return flip_xform_template.format(**format)


def get_cmtk_executable(tool: str) -> Path:
    path_str = os.environ.get("CMTK_DIR")
    if path_str is not None:
        out_path = Path(path_str) / tool
        if out_path.exists():
            return out_path
    paths = os.environ["PATH"].split(os.pathsep)
    for path_str in paths:
        path = Path(path_str)
        try:
            return next(path.glob(tool)).resolve()
        except StopIteration:
            continue
        except BaseException:
            raise
    raise FileNotFoundError("Could not find CMTK")


def validate_channels(channels: list[int | None]) -> tuple[int, ...]:
    """
    Takes list of channels and either returns the duplicated channels
    emits a warning if a channel is duplicated
    """
    not_none_channels = [c for c in channels if c is not None]
    if any(c < 1 for c in not_none_channels):
        raise ChannelValidationError("Channel 1 is the smallest possible channel")
    if len(not_none_channels) == len(set(not_none_channels)):
        return tuple()
    value_counts = pd.Series(not_none_channels).value_counts()
    assert value_counts.iloc[0] != 1
    logger.warning(f"Channel {value_counts.index[0]} is duplicated")
    dup_value_counts: pd.Series = value_counts.loc[value_counts != 1]
    return tuple(dup_value_counts.index.tolist())


def get_engine_with_context(ctx: dict, delete_previous=False) -> Engine:
    """
    gets the db engine
    """
    db_path = ctx.get("database")
    return get_engine(db_path, delete_previous=delete_previous)


def get_db_path():
    """
    gets the path of the database
    """
    # get path
    db_name = "template6alh.db"
    if os.name == "nt":  # Windows
        db_dir = (
            Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
            / "template6alh"
        )
    elif os.name == "posix":
        if "darwin" in os.uname().sysname.lower():  # macOS
            db_dir = Path.home() / "Library" / "template6alh"
        else:  # Linux/Unix
            db_dir = (
                Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
                / "template6alh"
            )
    else:
        db_dir = Path.home() / "logs"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / db_name


def get_engine(db_path: Path | None = None, delete_previous=False) -> Engine:
    """
    returns a sqlite engine, possibly destroying the first one and possiy
    finding a good path for it
    """
    if db_path is None:
        db_path = get_db_path()
    if delete_previous:
        if db_path.exists():
            logger.warning("Deleting old database")
        db_path.unlink(missing_ok=True)
    return create_engine(f"sqlite:///{str(db_path)}")


def get_logfile_path() -> Path:
    """
    Returns a pathlib Path object for a logfile, ensuring the path is suitable
    for the operating system (Windows, macOS, or Linux).

    Returns:
        Path: A pathlib Path pointing to the logfile.
    """
    logfile_name = "template6alh.log"
    if os.name == "nt":  # Windows
        log_dir = (
            Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
            / "template6alh"
            / "Logs"
        )
    elif os.name == "posix":
        if "darwin" in os.uname().sysname.lower():  # macOS
            log_dir = Path.home() / "Library" / "Logs" / "template6alh"
        else:  # Linux/Unix
            log_dir = (
                Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state"))
                / "template6alh"
                / "Logs"
            )
    else:
        log_dir = Path.home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / logfile_name


def write_nhdrs(header_dict: dict[str, dict], data: np.ndarray, folder: Path):
    """
    writes many nhdrs to the same data
    header_dict maps nhdr file names to metadata dicts
    data is the data to save
    folder is where all of the files go
    file names are defined by [folder / name for name in header_dict.keys()]
    """
    # write the data
    data_path = (folder / next(iter(header_dict.keys()))).with_suffix(".raw")
    # assert not data_path.exists()
    raw_data = data.tobytes(order="F")
    data_path.write_bytes(raw_data)
    # write headers
    dtype = data.dtype
    if dtype.byteorder == "=":  # Native endian
        endian = "little" if sys.byteorder == "little" else "big"
    else:  # Explicit endian
        endian = "little" if dtype.byteorder == "<" else "big"
    common_header = {
        "type": dtype.name,
        "endian": endian,
        "dimension": len(data.shape),
        "sizes": data.shape,
        "encoding": "raw",
        "data file": data_path.name,
    }
    for header_name, header in header_dict.items():
        # get header
        nrrd_header = common_header.copy()
        nrrd_header.update(header)
        # get file pointer
        if not header_name.endswith(".nhdr"):
            header_name += ".nhdr"
        with open(folder / header_name, "wb") as file:
            _write_header(file, nrrd_header)


def image_folders_from_file(
    image_folders: list[str], image_folders_file: str | None
) -> list[str] | None:
    """
    Gets image folders file or exits the program if both image_folder and
    image_folders_file are specified
    """
    if image_folders_file is None:
        if len(image_folders) == 0:
            return None
        return image_folders
    if len(image_folders) != 0:
        click.echo(
            "You must specify either image-folders or --image-folders-file (-f). Not both"
        )
        sys.exit(2)
    return Path(image_folders_file).read_text().split()


def get_spacings(metadata: dict) -> np.ndarray:
    directions = metadata.get("space directions")
    if directions is not None:
        return np.diag(directions)
    else:
        return metadata["spacings"]


def get_target_grid(path: Path) -> str:
    """
    reads a header from a nrrd file to get a target-grid string which is
    compatable with --target-grid option of reformatx
    """

    # Nx,Ny,Nz:dX,dY,dZ[:Ox,Oy,Oz] (dims:pixel:offset)
    def float_format(num) -> str:
        return f"{float(num):.4f}"

    header = nrrd.read_header(str(path))
    spacings = get_spacings(header)
    sizes = header["sizes"]
    return f"{','.join(str(s) for s in sizes)}:{','.join(float_format(s) for s in spacings)}"


def get_landmark_affine(src_landmarks: Path, dst_landmarks: Path, outpath: Path):
    """
    Does a rigid registration between landmarks to create a affine xform
    """
    run_with_logging(
        (
            get_cmtk_executable("fit_affine_xform_landmarks"),
            "--rigid",
            dst_landmarks,
            src_landmarks,
            outpath,
        )
    )
