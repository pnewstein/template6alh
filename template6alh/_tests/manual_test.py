"""
Some manual tests meant to be run as a script
"""

from pathlib import Path
from subprocess import run

import nrrd
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution

from template6alh import utils, matplotlib_slice


def main():
    folder_path = Path() / "data"
    folder_path.mkdir(exist_ok=True)

    header = {
        "space": "RAS",
        "sample units": ("micron", "micron", "micron"),
        "labels": ["Z", "Y", "X"],
    }
    in_coords = matplotlib_slice.CoordsSet(
        brain=(1, 1, 1), sez=(1, 3, 1), tip=(1, 3, 6), scale=np.array([0.5, 0.5, 0.5])
    )

    rotation_deg = [57.81790164, 126.92776522, -38.72899345]
    out_scale = np.array([0.4, 0.4, 0.4])
    rot = Rotation.from_euler("XYZ", rotation_deg, degrees=True)
    out_scaled = rot.apply(in_coords.to_array())
    out_shifted = out_scaled / out_scale
    out_array = (np.round(out_shifted) + np.stack([[1, 0, 2]] * 3)).astype(int)
    in_img = np.zeros((10, 10, 10)).astype(np.uint8)
    in_array = np.stack([in_coords.brain, in_coords.sez, in_coords.tip])
    fixed_path = folder_path / "fixed.nrrd"
    in_img[
        in_array[:, 0],
        in_array[:, 1],
        in_array[:, 2],
    ] = 255
    nrrd.write(
        str(fixed_path),
        in_img,
        {**header, **{"space directions": np.diag([0.5] * 3)}},
    )
    out_img = np.zeros((10, 10, 10)).astype(np.uint8)
    out_img[
        out_array[:, 0],
        out_array[:, 1],
        out_array[:, 2],
    ] = 255
    moving_path = folder_path / "moving.nrrd"
    nrrd.write(
        str(moving_path),
        out_img,
        {**header, **{"space directions": np.diag(out_scale)}},
    )
    moving_landmarks = moving_path.with_suffix(".landmarks")
    fixed_landmarks = fixed_path.with_suffix(".landmarks")
    # slicer = matplotlib_slice.write_landmarks(moving_path, moving_landmarks)
    # input()
    # slicer.quit()
    # slicer = matplotlib_slice.write_landmarks(fixed_path, fixed_landmarks)
    # input()
    # slicer.quit()
    landmark_affine = folder_path / "/landmark.xform"
    utils.get_landmark_affine(moving_landmarks, fixed_landmarks, landmark_affine)
    run(
        (
            utils.get_cmtk_executable("reformatx"),
            "--linear",
            "--outfile",
            folder_path / "reformated.nrrd",
            "--floating",
            moving_path,
            fixed_path,
            landmark_affine,
        )
    )


def realistic():
    folder = Path().home() / "Documents/t66alh/template"
    mask_tempalte = folder / "mask_template.nrrd"
    template_landmarks = folder / "template.landmarks"
    slicer = matplotlib_slice.write_landmarks(mask_tempalte, template_landmarks)
    # input()
    # slicer.quit()
    mask_landmarks = folder / "mask.landmarks"
    # slicer = matplotlib_slice.write_landmarks(mask_tempalte, mask_landmarks)
    # input()
    # slicer.quit()
    landmark_affine = folder / "landmark.xform"
    utils.get_landmark_affine(mask_landmarks, template_landmarks, landmark_affine)
    run(
        (
            utils.get_cmtk_executable("reformatx"),
            "--linear",
            "--outfile",
            folder / "reformated.nrrd",
            "--floating",
            mask_tempalte,
            mask_tempalte,
            landmark_affine,
        )
    )


def get_good_rotation():
    x = np.array([0.33333333, 0.4, 0.5])

    def objective(x):
        deg = x * 180

        rot = Rotation.from_euler("XYZ", deg, degrees=True)

        out_scale = np.array([0.4, 0.4, 0.4])
        out_array = rot.apply(in_coords.to_array())
        out_pix = out_array / out_scale
        centered = out_pix
        return np.abs(out_pix - np.round(out_pix)).sum()

    results = differential_evolution(objective, x0=x, bounds=[(-1, 1)] * 3)
