"""
tests functions
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import run
from logging import getLogger

import nrrd
import numpy as np

from template6alh.utils import (
    validate_channels,
    get_engine_with_context,
    write_nhdrs,
    get_target_grid,
    run_with_logging,
    get_cmtk_executable,
)
from template6alh.matplotlib_slice import CoordsSet
from template6alh.iterative_shape_averaging import ZERO_INIT as NO_FLIP

getLogger().setLevel("INFO")


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
        low_res = folder / "low_res.nrrd"
        data = np.ones((10, 3, 100)).astype(np.uint8)
        header = {"spacings": [1, 2, 0.1]}
        nrrd.write(str(low_res), data=data, header=header)
        high_res = folder / "high_res.nrrd"
        header = {"spacings": [0.5, 1, 0.05]}
        data = np.zeros((3, 3, 3)).astype(np.uint8)
        nrrd.write(str(high_res), data=data, header=header)
        target_grid = get_target_grid(high_res, low_res, False)
        assert target_grid == "20,6,200:0.5000,1.0000,0.0500"
        xform = folder / "none.xform"
        xform.write_text(NO_FLIP)
        run(
            (
                get_cmtk_executable("reformatx"),
                "-o",
                folder / "out.nrrd",
                "--target-grid",
                target_grid,
                "--floating",
                low_res,
                xform,
            )
        )
        data, md = nrrd.read(folder / "out.nrrd")
        assert np.allclose(np.diag(md["space directions"]), header["spacings"])
        assert data.mean() > 0.5
        # This time test output
        fine = folder / "fine.nrrd"
        fine_header = {
            "sizes": [28, 45, 25],
            "space directions": [
                (0.070600003004074097, 0, 0),
                (0, 0.070600003004074097, 0),
                (0, 0, 0.070600003004074097),
            ],
        }
        fine_data = np.zeros(shape=tuple(fine_header["sizes"]))
        nrrd.write(str(fine), data=fine_data, header=fine_header)
        corse = folder / "corse.nrrd"
        corse_header = {
            "sizes": [9, 45, 25],
            "space directions": [
                (0.5, 0, 0),
                (0, 0.5, 0),
                (0, 0, 0.5),
            ],
        }
        corse_data = np.zeros(shape=tuple(corse_header["sizes"]))
        nrrd.write(str(corse), data=corse_data, header=corse_header)
        target_grid = get_target_grid(fine, corse)
        assert target_grid == "64,319,178:0.0706,0.0706,0.0706"


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
