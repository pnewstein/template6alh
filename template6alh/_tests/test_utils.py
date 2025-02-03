"""
tests functions
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import run

import nrrd
import numpy as np

from template6alh.utils import (
    validate_channels,
    get_engine_with_context,
    write_nhdrs,
    get_flip_xform,
    get_target_grid,
    run_with_logging,
    get_cmtk_executable,
)
from template6alh.matplotlib_slice import CoordsSet


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
        assert get_target_grid(path) == "10,3,100:1.0000,2.0000,0.1000"
        header = {"space directions": np.diag([2, 20, 0.1])}
        nrrd.write(str(path), data=data, header=header)
        assert get_target_grid(path) == "10,3,100:2.0000,20.0000,0.1000"


def mat_to_array(mat: bytes):
    return np.array([[float(n) for n in l.split()] for l in mat.decode().splitlines()])


def test_coords_set():
    c1 = (0, 1, 2)
    c2 = (0.0, 1.0, 3.0)
    c3 = (0.0, 3.0, 3.0)
    c1p1 = (1.0, 1.0, 2.0)
    c2p1 = (1.0, 1.0, 3.0)
    c3p1 = (1.0, 3.0, 3.0)
    scale = np.array([1, 1, 1])
    coords_set = CoordsSet(brain=c1, sez=c2, tip=c3, scale=scale)
    assert coords_set.to_dict() == {"brain": c1, "sez": c2, "tip": c3}
    with TemporaryDirectory() as folder_str:
        folder = Path(folder_str)
        file = folder / "landmarks.txt"
        file.write_text(coords_set.to_cmtk())
        p1 = folder / "plus1.txt"
        out = folder / "landmark.xform"
        p1.write_text(CoordsSet(brain=c1p1, sez=c2p1, tip=c3p1, scale=scale).to_cmtk())
        run_with_logging(
            (
                get_cmtk_executable("fit_affine_xform_landmarks"),
                "--rigid",
                file,
                file,
                out,
            )
        )
        result = run_with_logging((get_cmtk_executable("dof2mat"), out))
        mat = mat_to_array(result.stdout)
        assert np.allclose(mat, np.eye(4))
        run_with_logging(
            (
                get_cmtk_executable("fit_affine_xform_landmarks"),
                "--rigid",
                p1,
                file,
                out,
            )
        )
        result = run_with_logging((get_cmtk_executable("dof2mat"), out))
        mat = mat_to_array(result.stdout)
        expected = np.eye(4)
        expected[3, 0] = -1
        assert np.allclose(mat, expected)
