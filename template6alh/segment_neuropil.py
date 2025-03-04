"""
This code preforms a segmentation of the neuropil after significanly lowering
the resolution
"""

import logging

import pandas as pd
import numpy as np
from skimage import morphology, restoration, filters, measure, measure, exposure
from scipy import ndimage as ndi

logger = logging.getLogger(__name__)

default_args_make_neuropil_mask = {
    "new_scale": 0.5,
    "gamma": 1.0,
    "opening_size": 2.0,
}


def make_neuropil_mask(
    img_data: np.ndarray,
    scale: tuple[float, float, float],
    new_scale: float | None,
    gamma: float,
    opening_size: float,
) -> np.ndarray:
    """
    does image morphology manipulations to isolate the neuropils
    returns the processed image
    """

    assert len(img_data.shape) == 3
    if new_scale is not None:
        img_data = ndi.zoom(img_data, np.array(scale) / new_scale)
        scale = (new_scale, new_scale, new_scale)
    img_data = img_data * (np.iinfo(img_data.dtype).max / img_data.max())
    img_data = exposure.adjust_gamma(img_data, gamma)
    size = tuple(opening_size / e for e in scale)
    kernel = restoration.ellipsoid_kernel(size, 1) != np.inf
    opened = morphology.opening(img_data, kernel)
    threshold = filters.threshold_otsu(opened)
    neuropil_mask = opened > threshold
    # ensure there is only one neuropil
    labeled = measure.label(neuropil_mask)
    assert isinstance(labeled, np.ndarray)
    value_counts = pd.Series(labeled.ravel()).value_counts()
    npix_neuropils = value_counts.drop(0)
    frac_neuropils = npix_neuropils / npix_neuropils.sum()
    assert isinstance(frac_neuropils, pd.Series)
    frac_neuropils.sort_values(inplace=True, ascending=False)
    biggest_neuropil = frac_neuropils.iloc[0] * 100
    if biggest_neuropil > 95:
        logger.info(
            f"Neuropil takes up {biggest_neuropil:.0f}% of the large bright objects"
        )
    elif biggest_neuropil > 85:
        logger.debug(
            f"Neuropil takes up {biggest_neuropil:.0f}% of the large bright objects"
        )
    else:
        logger.warning(
            f"Neuropil takes up only {biggest_neuropil:.0f}% of the large bright objects"
        )
    bad_inds = frac_neuropils.iloc[1:].index
    neuropil_mask[np.isin(labeled, bad_inds)] = 0
    return neuropil_mask.astype(np.uint8)
