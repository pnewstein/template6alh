"""
code for reading raw data

all generators have the same signature yeilding each image one a time
each image is expected to be in CZYX form
"""

from pathlib import Path
from typing import Generator, Any
from aicspylibczi import CziFile
from skimage.data import binary_blobs

import numpy as np


def read_test(path: Path) -> Generator[tuple[dict[str, Any], np.ndarray], None, None]:
    """
    generates test data in the same manner as read_czi
    """
    _ = path
    metadata = {l: 1 for l in "CZYX"}
    for _ in range(2):
        volume = binary_blobs(
            length=50, blob_size_fraction=0.1, n_dim=3, volume_fraction=0.1
        )
        yield metadata, np.stack([volume] * 3).astype(np.uint8)


def read_czi(path: Path) -> Generator[tuple[dict[str, Any], np.ndarray], None, None]:
    """
    reads a czi file
    generates metadata and 4d images CZYX
    metadata is guaranteed to have keys "X", "Y", "Z"

    """
    czi = CziFile(path)
    metadata = czi.meta
    assert metadata is not None
    distances = metadata.find("./Metadata/Scaling/Items")
    assert distances is not None
    # get xyz scale
    spacings_dict: dict[str, float] = {}
    for distance in distances:
        value = distance.find("Value")
        assert value is not None
        assert value.text is not None
        spacings_dict[next(iter(distance.items()))[-1]] = float(value.text) * 10e5
    # iterate through images
    assert czi.shape_is_consistent
    (dimensions,) = czi.get_dims_shape()
    assert all(i[0] == 0 for i in dimensions.values())
    image_spec_range_max = {
        k: end - start for k, (start, end) in dimensions.items() if k not in "XYZC"
    }
    for index in np.ndindex(tuple(image_spec_range_max.values())):
        image_spec = {lbl: ind for lbl, ind in zip(image_spec_range_max.keys(), index)}
        image, dims = czi.read_image(**image_spec)
        assert "".join(dim for dim, count in dims if count != 1) == "CZYX"
        yield spacings_dict, image.squeeze()
