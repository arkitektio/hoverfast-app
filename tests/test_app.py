"""The app for itself, with no server.

The tiling and the splitting of nuclei are tested with a stand-in for the model
that sees dark discs as nuclei, the model itself with its pretrained weights, and
`local_runtime` starts the app as a run starts it. Run these with
`uv run --extra cpu pytest`; the release workflow does too.

The action stores its results, so calling it needs the service: that test is in
`test_app_hub.py`.
"""

import hashlib

import numpy as np
import pytest
import scipy.ndimage as ndi
from arkitekt.runtime import Runtime
from hoverfast.wsi.postprocess import pre_watershed

from app import app, segment_at_model_scale
from hoverfast_app.measure import nucleus_table
from hoverfast_app.segment import region_labels, segment, to_uint8_rgb


def hv_maps(labels: np.ndarray) -> np.ndarray:
    """Horizontal and vertical maps as HoverFast is trained on them: -1 to 1 across each nucleus."""
    maps = np.zeros((2, *labels.shape), np.float32)
    for index, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        inside = labels[box] == index
        ys, xs = np.mgrid[box]
        for channel, coords in enumerate((xs, ys)):
            values = coords[inside].astype(np.float32)
            low, high = values.min(), values.max()
            maps[channel][box][inside] = (values - low) / max(high - low, 1) * 2 - 1
    return maps


def fake_predict(regions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A stand-in for the model: dark pixels are nuclei, and every connected dark part is one."""
    masks = regions.mean(axis=-1) < 128
    return masks, np.stack([hv_maps(ndi.label(mask)[0]) for mask in masks])


def disc(shape: tuple[int, int], cy: float, cx: float, r: float) -> np.ndarray:
    ys, xs = np.ogrid[: shape[0], : shape[1]]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= r**2


def image_with_discs(shape: tuple[int, int], centres: list[tuple[int, int]], r: int = 10) -> np.ndarray:
    """White, with a black disc of radius `r` at each centre."""
    rgb = np.full((*shape, 3), 255, np.uint8)
    for cy, cx in centres:
        rgb[disc(shape, cy, cx, r)] = 0
    return rgb


# --- splitting one region into nuclei ---


def test_two_touching_nuclei_are_split_in_two():
    shape = (64, 96)
    truth = np.zeros(shape, np.int32)
    truth[disc(shape, 32, 34, 14)] = 1
    truth[disc(shape, 32, 60, 14) & (truth == 0)] = 2
    mask = truth > 0
    assert ndi.label(mask)[1] == 1, "the two discs touch"

    labels = region_labels(mask, *pre_watershed(mask, hv_maps(truth)), min_area_px=20)

    assert len(np.unique(labels[labels > 0])) == 2
    assert labels[32, 30] != labels[32, 64]


def test_a_nucleus_smaller_than_the_least_area_is_left_out():
    shape = (64, 64)
    truth = (disc(shape, 20, 20, 12) | disc(shape, 48, 48, 3)).astype(np.int32)
    truth = ndi.label(truth)[0]
    mask = truth > 0

    labels = region_labels(mask, *pre_watershed(mask, hv_maps(truth)), min_area_px=100)

    assert labels.max() == 1
    assert labels[20, 20] == 1 and labels[48, 48] == 0


# --- tiling ---


def test_every_nucleus_is_found_once_across_the_tiles():
    shape = (1500, 777)
    # One on the seam between the first two rows of tiles, one in their overlap, one
    # cut by each of the image's edges, and one in the middle of nowhere.
    centres = [(896, 400), (950, 200), (3, 400), (700, 3), (1495, 600), (300, 773), (300, 300)]
    rgb = image_with_discs(shape, centres)

    labels = segment(rgb, fake_predict, min_area_px=20)

    assert labels.shape == shape
    ids = [labels[cy, cx] for cy, cx in centres]
    assert all(ids) and len(set(ids)) == len(centres)
    assert labels.max() == len(centres)
    for (cy, cx), label in zip(centres, ids):
        assert np.all(labels[disc(shape, cy, cx, 9)] == label), "a nucleus is not cut at a seam"


def test_an_image_smaller_than_a_tile_is_one_tile():
    tiles = []
    labels = segment(image_with_discs((200, 300), [(100, 150)]), fake_predict, min_area_px=20, on_tile=lambda d, t: tiles.append((d, t)))

    assert labels.shape == (200, 300)
    assert labels.max() == 1
    assert tiles == [(1, 1)]


def test_an_image_without_nuclei_has_no_labels():
    labels = segment(np.full((300, 300, 3), 255, np.uint8), fake_predict, min_area_px=20)

    assert not labels.any()


def test_tiles_are_predicted_in_batches():
    seen = []

    def counting_predict(regions):
        seen.append(len(regions))
        return fake_predict(regions)

    segment(image_with_discs((2000, 2000), [(1000, 1000)]), counting_predict, min_area_px=20, batch=4)

    assert seen == [4, 4, 1]  # three rows and three columns of tiles


def test_an_image_of_larger_pixels_is_labelled_on_its_own_grid():
    rgb = image_with_discs((300, 400), [(150, 200)], r=6)

    labels = segment_at_model_scale(rgb, fake_predict, pixel_size_um=0.5, min_area_um2=5)

    assert labels.shape == (300, 400)
    assert labels[150, 200] == 1


# --- pixels ---


def test_rgb_pixels_are_brought_to_uint8():
    assert to_uint8_rgb(np.ones((2, 2, 3), np.uint8)).dtype == np.uint8
    assert to_uint8_rgb(np.full((2, 2, 3), 0.5, np.float32)).max() == 128
    assert to_uint8_rgb(np.array([[[0, 1000, 4000, 7]]], np.uint16)).tolist() == [[[0, 64, 255]]]
    with pytest.raises(ValueError, match="three channels"):
        to_uint8_rgb(np.zeros((4, 4)))


# --- measurements ---


def test_the_table_has_a_row_per_nucleus():
    labels = np.zeros((40, 40), np.int32)
    labels[2:6, 2:7] = 1
    labels[20:30, 20:30] = 3

    table = nucleus_table(labels, pixel_size_um=0.5)

    assert table["object_id"].tolist() == [1, 3]
    assert table["area_px"].tolist() == [20, 100]
    assert table["area_um2"].tolist() == [5.0, 25.0]
    assert len({len(column) for column in table.values()}) == 1


# --- the model ---


def test_the_weights_are_the_published_ones():
    from hoverfast_app.model import DEFAULT_WEIGHTS, WEIGHTS_SHA256

    assert hashlib.sha256(DEFAULT_WEIGHTS.read_bytes()).hexdigest() == WEIGHTS_SHA256


def test_the_model_predicts_a_mask_and_maps_as_large_as_the_region():
    import torch

    from hoverfast_app.model import load, predict

    model = load(device=torch.device("cpu"))
    assert next(model.parameters()).dtype == torch.float32

    mask, maps = predict(np.random.default_rng(1).integers(0, 255, (1, 333, 517, 3), dtype=np.uint8), model)

    assert mask.shape == (1, 333, 517) and mask.dtype == bool
    assert maps.shape == (1, 2, 333, 517) and maps.dtype == np.float32


# --- the app ---


def test_the_app_starts_and_offers_segment_nuclei(local_runtime: Runtime[None]):
    assert local_runtime.app is app
    assert "segment_nuclei" in app.registry.implementations
