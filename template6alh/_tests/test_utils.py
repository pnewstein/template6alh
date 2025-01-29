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
    get_target_grid,
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


def test_get_target_grid():
    with TemporaryDirectory() as folder_str:
        folder = Path(folder_str)
        path = folder / "out.nrrd"
        data = np.zeros((10, 3, 100))
        header = {"spacings": [1, 2, 0.1]}
        nrrd.write(str(path), data=data, header=header)
        assert get_target_grid(path) == "10,3,100:1.0,2.0,0.1"
        header = {"space directions": np.diag([2, 20, 0.1])}
        nrrd.write(str(path), data=data, header=header)
        assert get_target_grid(path) == "10,3,100:2.0,20.0,0.1"
