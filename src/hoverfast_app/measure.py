"""What is measured of every nucleus: one row per label, keyed by its id."""

import numpy as np
from skimage.measure import regionprops_table

_PROPERTIES = ("label", "area", "centroid", "bbox", "eccentricity", "solidity")


def nucleus_table(labels: np.ndarray, pixel_size_um: float) -> dict[str, np.ndarray]:
    """The columns of the table of the nuclei in (y, x) `labels`, of pixels `pixel_size_um` wide.

    Positions and extents are in pixels of `labels`; the area is also in µm².
    """
    props = regionprops_table(labels, properties=_PROPERTIES)
    area = props["area"].astype(np.int64)
    return {
        "object_id": props["label"].astype(np.int64),
        "area_px": area,
        "area_um2": area * pixel_size_um**2,
        "centroid_y": props["centroid-0"].astype(np.float64),
        "centroid_x": props["centroid-1"].astype(np.float64),
        "bbox_min_y": props["bbox-0"].astype(np.int64),
        "bbox_min_x": props["bbox-1"].astype(np.int64),
        "bbox_max_y": props["bbox-2"].astype(np.int64),
        "bbox_max_x": props["bbox-3"].astype(np.int64),
        "eccentricity": props["eccentricity"].astype(np.float64),
        "solidity": props["solidity"].astype(np.float64),
    }
