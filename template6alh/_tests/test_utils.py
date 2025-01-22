"""
tests functions
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import nrrd
import numpy as np

from template6alh.utils import (
    validate_channels,
    get_engine_with_context,
    write_nhdrs,
    get_flip_xform,
    get_init_xform,
    run_with_logging,
    get_cmtk_executable,
)


def test_get_flip_xform():
    string = get_flip_xform("011")
    assert "-1" in string
    assert "scale 1" in string


def test_validate_channels():
    chans = validate_channels([1, 2, None])
    assert chans == tuple()
    chans = validate_channels([None, None, None])
    assert chans == tuple()
    chans = validate_channels([1, 1, None])
    assert chans == (1,)


def test_get_engine_with_context():
    assert get_engine_with_context({"database": "out"}).url.database == "out"


def test_write_nhdrs():
    rng = np.random.default_rng(100)
    data = rng.integers(0, 255, size=(10, 10, 10), dtype=np.uint8)
    with TemporaryDirectory() as folder_str:
        folder = Path(folder_str)
        write_nhdrs({"out": {}}, data, folder)
        read_data, _ = nrrd.read(folder / "out.nhdr")
        assert np.array_equal(read_data, data)


def test_get_init_xform():
    template_data = np.stack(
        [
            [
                [0, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
            ]
        ]
        * 3
    ).astype(np.uint8)
    floating_data = np.stack(
        [
            [
                [0, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
            ]
        ]
        * 3
    ).astype(np.uint8)
    folder = Path("/tmp/tdir")
    folder.mkdir(exist_ok=True)
    template_path = folder / "template.nrrd"
    header = dict(
        [
            ("space", "right-anterior-superior"),
            ("space directions", [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
            ("labels", ["x", "y", "z"]),
        ]
    )
    nrrd.write(str(template_path), template_data, header=header)
    flip_path = folder / "flip.nrrd"
    flip_header = dict(
        [
            ("space", "right-anterior-superior"),
            ("space directions", [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -0.5]]),
            # ("spacings", [.5, .5, -.5]),
            ("labels", ["x", "y", "z"]),
        ]
    )
    nrrd.write(str(flip_path), floating_data, header=flip_header)
    assert False
    init_xform_path = folder / "affine.xform"
    # get_init_xform(flip_path, template_path, "001", init_xform_path)
    reformat_path = folder / "reformat.nrrd"
    run_with_logging(
        (
            get_cmtk_executable("reformatx"),
            "-o",
            reformat_path,
            "--floating",
            flip_path,
            template_path,
            init_xform_path,
        )
    )
    data, _ = nrrd.read(str(reformat_path))
    print(data)
    assert data.sum() != 0
    assert False

