"""Post-processing for *predicted* segmentation masks -- raster class-index arrays, as
opposed to helpers/mask_rasterize.py which builds ground-truth masks from vector polygons.
"""

import numpy as np
from scipy import ndimage

from helpers.mask_rasterize import CLASS_BACKGROUND


def filter_small_regions(
    mask: np.ndarray,
    class_id: int,
    gsd_m: float,
    min_area_m2: float,
    background_class: int = CLASS_BACKGROUND,
) -> np.ndarray:
    """Reclassify connected regions of `class_id` smaller than `min_area_m2` back to
    `background_class`.

    Safety net for the model's *prediction*, not the primary size filter: training masks
    already only label parkings >=1500m^2 (loi APER threshold, `MIN_PARKING_AREA_M2` in
    `helpers.mask_rasterize`) since dropping that filter to widen the training signal was
    tried and measured worse, not better (see docs/roadmap_segmentation.md §8, item 9) --
    small parkings don't have a clean visual signature at 10m/pixel, so labeling them
    Parking added noise rather than useful examples. This function still catches the rare
    spurious small blob the model predicts despite that, so a reported result never counts
    a sub-threshold detection as a real, business-relevant (loi APER-eligible) parking.

    @param gsd_m: ground sample distance (meters/pixel) of `mask`, to convert connected
    component pixel counts into real-world area.
    """
    result = mask.copy()
    class_pixels = mask == class_id
    labeled, n_components = ndimage.label(class_pixels)
    if n_components == 0:
        return result

    pixel_area_m2 = gsd_m**2
    component_sizes = ndimage.sum(class_pixels, labeled, index=range(1, n_components + 1))
    small_component_ids = [i + 1 for i, size in enumerate(component_sizes) if size * pixel_area_m2 < min_area_m2]

    if small_component_ids:
        result[np.isin(labeled, small_component_ids)] = background_class

    return result
