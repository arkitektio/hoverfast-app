"""The app against the services it uses: its action, storing real labels and tables.

`teststack` logs the app in to a hub made for the tests. It needs Docker and
konstruktor, and about a minute to start, so these are not part of a bare
`uv run --extra cpu pytest`: every test here is marked `hub`. Run them with
`uv run --extra cpu pytest -m hub`.

What needs no server is tested in `test_app.py`.
"""

import numpy as np
import pytest
import xarray as xr
from arkitekt.testing import TestStack
from mikro import Mikro
from mikro.api.schema import AxisInput, AxisType

# Every test in this file runs against a hub.
pytestmark = pytest.mark.hub


def he_like_image(size: int = 256) -> np.ndarray:
    """Pink, as eosin stains, with purple discs where nuclei are, as haematoxylin stains them."""
    rgb = np.empty((size, size, 3), np.uint8)
    rgb[:] = (235, 170, 200)
    ys, xs = np.ogrid[:size, :size]
    for cy, cx in [(60, 60), (60, 180), (150, 110), (200, 200)]:
        rgb[(ys - cy) ** 2 + (xs - cx) ** 2 <= 14**2] = (90, 50, 130)
    return rgb


def upload_rgb(mikro: Mikro, rgb: np.ndarray, name: str):
    return mikro.create_array_dataset(
        data=xr.DataArray(rgb, dims=("y", "x", "c")),
        scales=[],
        name=name,
        axes=[
            AxisInput(name="y", type=AxisType.SPACE),
            AxisInput(name="x", type=AxisType.SPACE),
            AxisInput(name="c", type=AxisType.CHANNEL),
        ],
    )


def test_the_nuclei_of_an_image_are_labelled_and_measured(teststack: TestStack[None]):
    mikro = teststack.runtime.require(Mikro)
    image = upload_rgb(mikro, he_like_image(), "H&E")

    labels, table = teststack.call("segment_nuclei", image=image.lens(), pixel_size_um=0.25)

    pixels = labels.data.squeeze().compute().values
    assert pixels.shape == (256, 256)
    ids = np.unique(pixels[pixels > 0])
    assert len(ids) >= 1, "the nuclei are found"
    rows = mikro.get_table_dataset(table.table_dataset.id).data.df()
    assert sorted(rows["object_id"].tolist()) == ids.tolist()


def test_an_image_without_nuclei_gives_empty_labels_and_an_empty_table(teststack: TestStack[None]):
    mikro = teststack.runtime.require(Mikro)
    image = upload_rgb(mikro, np.full((128, 128, 3), (235, 170, 200), np.uint8), "Background")

    labels, table = teststack.call("segment_nuclei", image=image.lens())

    assert not labels.data.squeeze().compute().values.any()
    assert len(mikro.get_table_dataset(table.table_dataset.id).data.df()) == 0
