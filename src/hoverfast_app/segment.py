"""From an RGB image of any size to the instance labels of its nuclei.

The model sees the image in overlapping tiles, as upstream's whole-slide pipeline
does: tiles of 1024 pixels that overlap by 128. Each tile is split into nuclei on
its own, by HoverFast's own watershed (`pre_watershed`, `watershed_object_roi`),
and keeps a nucleus only when its centroid lies in the tile's core. The cores tile
the image without overlap, so a nucleus on a seam is kept by exactly one tile.

What predicts is passed in (`hoverfast_app.model.predict`, bound to a model), so
the tiling is tested without one.
"""

from collections.abc import Callable

import cv2
import numpy as np
import scipy.ndimage as ndi
from hoverfast.roi.pipeline import watershed_object_roi
from hoverfast.wsi.postprocess import pre_watershed
from skimage.measure import regionprops

Predict = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
"""(N, H, W, 3) uint8 regions -> (N, H, W) bool nucleus mask, (N, 2, H, W) float maps."""


def to_uint8_rgb(pixels: np.ndarray) -> np.ndarray:
    """A (y, x, c) image as the uint8 RGB the model was trained on.

    uint8 is taken as it is, floats up to 1 as fractions of full intensity, and
    anything else is stretched from its minimum to its maximum.
    """
    if pixels.ndim != 3 or pixels.shape[-1] < 3:
        raise ValueError(f"An RGB image has three channels last, and this one is {pixels.shape}")
    pixels = pixels[..., :3]
    if pixels.dtype == np.uint8:
        return np.ascontiguousarray(pixels)
    fractions = np.issubdtype(pixels.dtype, np.floating)
    pixels = pixels.astype(np.float64)
    low, high = float(pixels.min()), float(pixels.max())
    if fractions and low >= 0 and high <= 1:
        low, high = 0.0, 1.0
    if high <= low:
        return np.zeros(pixels.shape, np.uint8)
    return np.round((pixels - low) / (high - low) * 255).astype(np.uint8)


def region_labels(
    mask: np.ndarray, dist: np.ndarray, marker: np.ndarray, opening: np.ndarray, min_area_px: float
) -> np.ndarray:
    """The nuclei of one region as int32 labels, split as upstream's ROI pipeline splits them.

    This is `hoverfast.roi.pipeline.region_feature_roi` with its files left out:
    every connected part of the mask is watershed into nuclei, and each nucleus
    larger than `min_area_px` is filled in under its own id.
    """
    labels = np.zeros(mask.shape, np.int32)
    next_id = 1
    for rg in regionprops(ndi.label(mask)[0]):
        if rg.area < min_area_px:
            continue
        ymin, xmin, ymax, xmax = rg.bbox
        contours = watershed_object_roi(
            rg,
            dist[ymin:ymax, xmin:xmax],
            marker[ymin:ymax, xmin:xmax] * rg.image,
            opening[ymin:ymax, xmin:xmax],
            (xmin, ymin),
            0,  # no polygon simplification: the labels follow the pixels
            min_area_px,
        )
        for contour in contours:
            cv2.fillPoly(labels, [contour.reshape(-1, 1, 2).astype(np.int32)], next_id)
            next_id += 1
    return labels


def _tiling(length: int, tile: int, margin: int) -> tuple[int, int, int]:
    """The tile length, step and count along an axis of `length` pixels."""
    size = min(tile, -(-(length + margin) // 4) * 4)  # a small image is one tile, of a multiple of 4
    step = size - margin
    return size, step, -(-length // step)


def segment(
    rgb: np.ndarray,
    predict: Predict,
    *,
    min_area_px: float,
    tile: int = 1024,
    margin: int = 128,
    batch: int = 1,
    on_tile: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """The nuclei of a (y, x, 3) uint8 image as int32 labels, 0 where there is none.

    Tiles are `tile` pixels (fewer for a smaller image) and overlap by `margin`;
    `batch` of them are predicted at once. `on_tile(done, total)` is called after
    every batch.
    """
    height, width, _ = rgb.shape
    if margin % 2 or margin >= tile:
        raise ValueError("The margin has to be even and smaller than a tile")
    tile_h, step_h, rows = _tiling(height, tile, margin)
    tile_w, step_w, cols = _tiling(width, tile, margin)
    half = margin // 2

    # Each tile's core is the `step` pixels after its first `half`: the cores of the
    # tiles are the image, once. Mirrored around, so the first tile starts at -half.
    padded = np.pad(
        rgb,
        ((half, rows * step_h + margin - half - height), (half, cols * step_w + margin - half - width), (0, 0)),
        mode="symmetric",
    )

    labels = np.zeros((height, width), np.int32)
    next_id = 1
    origins = [(r * step_h, c * step_w) for r in range(rows) for c in range(cols)]
    for start in range(0, len(origins), batch):
        chunk = origins[start : start + batch]
        regions = np.stack([padded[y : y + tile_h, x : x + tile_w] for y, x in chunk])
        masks, maps = predict(regions)
        for (y, x), mask, tile_maps in zip(chunk, masks, maps):
            # The model saw the mirror around the image, as context; its nuclei are not
            # the image's. Left in, a nucleus cut by the edge would merge with its own
            # mirror image, and be dropped for a centroid outside the image.
            outside = np.ones(mask.shape, bool)
            outside[max(half - y, 0) : height + half - y, max(half - x, 0) : width + half - x] = False
            mask = mask & ~outside
            dist, marker, opening = pre_watershed(mask, tile_maps)
            if dist is None or marker is None or opening is None:
                continue  # no nucleus in this tile
            for nucleus in regionprops(region_labels(mask, dist, marker, opening, min_area_px)):
                cy, cx = nucleus.centroid
                if not (half <= cy < half + step_h and half <= cx < half + step_w):
                    continue
                # Tile pixel (i, j) is image pixel (y + i - half, x + j - half).
                coords = nucleus.coords + (y - half, x - half)
                inside = (coords[:, 0] >= 0) & (coords[:, 0] < height) & (coords[:, 1] >= 0) & (coords[:, 1] < width)
                ys, xs = coords[inside].T
                free = labels[ys, xs] == 0
                if free.any():
                    labels[ys[free], xs[free]] = next_id
                    next_id += 1
        if on_tile:
            on_tile(min(start + batch, len(origins)), len(origins))
    return labels
