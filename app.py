"""HoverFast for Arkitekt: segments the nuclei of an H&E image.

HoverFast (https://github.com/choosehappy/HoverFast, BSD-3, Copyright (c) 2024,
Andrew Janowczyk) is a small multi-scale U-Net that predicts where nuclei are and,
for every pixel, how far it is from its nucleus's centre; a watershed splits
touching nuclei along those maps. The app runs its pretrained cross-tissue model
on an RGB image in mikro, tile by tile, and returns the nuclei as a label mask on
the image's grid and a table of their measurements, keyed by the mask's ids.

Run:  arkitekt run dev
"""

import functools
from typing import Annotated

import cv2
import numpy as np
import xarray as xr
from arkitekt import App, Description, DescriptorOperator, Provides, Requires, Task, run
from mikro import Mikro, mikro_service
from mikro.api.schema import (
    ArrayLens,
    AxisType,
    ColumnInput,
    ColumnRole,
    DatasetIdentifiesInput,
    TableLens,
    ValueRelation,
)
from mikro.arkitekt.specs import (
    N_CHANNEL_AXES,
    N_CHANNELS,
    N_SPACE_AXES,
    N_TIMEPOINTS,
    VALUE_KIND,
)

from hoverfast_app.measure import nucleus_table
from hoverfast_app.segment import Predict, segment, to_uint8_rgb

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.2.1"

# The app is a declaration: who it is, what it offers, and the services it uses.
# `arkitekt run` finds it in this module (as `app`) and runs it; so does
# `python app.py`, below.
app = App(
    "hoverfast-app",
    __version__,
    description="HoverFast nuclei segmentation of H&E images",
    author="jhnnsrs",
    services=[mikro_service],
)

COLLECTIONS = ["segmentation", "nuclei", "histology"]

# Written out as plain assignments of one Annotated each: a spec alias nested in
# another Annotated loses its markers. An argument says what it requires, a return
# what it provides.
StillRGBImage = Annotated[
    ArrayLens,
    Requires(key=N_SPACE_AXES, operator=DescriptorOperator.EQUALS, value=2),
    Requires(key=N_CHANNEL_AXES, operator=DescriptorOperator.EQUALS, value=1),
    Requires(key=N_CHANNELS, operator=DescriptorOperator.EQUALS, value=3),
    Requires(key=N_TIMEPOINTS, operator=DescriptorOperator.LTE, value=1),
]
"""An RGB plane with at most one timepoint."""

LabelImage = Annotated[
    ArrayLens,
    Provides(key=N_SPACE_AXES, operator=DescriptorOperator.EQUALS, value=2),
    Provides(key=VALUE_KIND, operator=DescriptorOperator.EQUALS, value="categorical"),
]
"""A plane of object ids."""

# The pixel size the model was trained at, in µm, and how far off an image's may be
# (as a fraction) before it is resampled to it.
MODEL_PIXEL_SIZE_UM = 0.25
RESAMPLE_TOLERANCE = 0.1


@functools.lru_cache(maxsize=1)
def _model() -> tuple[Predict, int]:
    """The pretrained model, loaded on the first segmentation rather than at start.

    torch is imported here too, so declaring the app does not wait for it. Returns
    what predicts, and how many tiles to predict at once: four on a GPU, one on a CPU.
    """
    from hoverfast_app import model

    loaded = model.load()
    on_gpu = next(loaded.parameters()).device.type == "cuda"
    return functools.partial(model.predict, model=loaded), 4 if on_gpu else 1


def segment_at_model_scale(
    rgb: np.ndarray,
    predict: Predict,
    pixel_size_um: float,
    min_area_um2: float,
    **kwargs,
) -> np.ndarray:
    """Segment a (y, x, 3) uint8 image of `pixel_size_um` pixels at the scale the model was trained at.

    An image of other pixels is resized to the model's for the prediction, and its
    labels are resized back, so they are on the image's own grid.
    """
    if pixel_size_um <= 0:
        raise ValueError("The pixel size has to be positive")
    height, width, _ = rgb.shape
    factor = pixel_size_um / MODEL_PIXEL_SIZE_UM
    resample = abs(factor - 1) > RESAMPLE_TOLERANCE
    working_size_um = MODEL_PIXEL_SIZE_UM if resample else pixel_size_um
    if resample:
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, size, interpolation=interpolation)
    labels = segment(rgb, predict, min_area_px=min_area_um2 / working_size_um**2, **kwargs)
    if resample:
        labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
    return labels


@app.action(name="Segment Nuclei (HoverFast)", collections=COLLECTIONS, concurrency="serial")
def segment_nuclei(
    image: Annotated[StillRGBImage, Description("The H&E image to segment, in RGB")],
    pixel_size_um: Annotated[
        float, Description("How wide a pixel of the image is, in µm. The model was trained at 0.25 (40x)")
    ] = 0.25,
    min_area_um2: Annotated[float, Description("Nuclei smaller than this, in µm², are left out")] = 5.0,
    *,
    mikro: Mikro,
    task: Task,
) -> tuple[LabelImage, TableLens]:
    """Segment Nuclei (HoverFast)

    Segments every nucleus of an H&E image with HoverFast's pretrained cross-tissue
    model. An image of other than 0.25 µm pixels is resampled to them for the model,
    and its labels back to the image.

    Returns
    -------
    LabelImage
        The nuclei, each under its own id, on the image's grid
    TableLens
        One row per nucleus, keyed by its id in the labels: its area, centroid,
        bounding box, eccentricity and solidity
    """
    data = image.data
    space_axes = image.axes_of_type(AxisType.SPACE)
    (channel_axis,) = image.axes_of_type(AxisType.CHANNEL)
    plane = data.squeeze([d for d in data.dims if d not in space_axes and d != channel_axis])

    task.progress(5, "Reading image")
    rgb = to_uint8_rgb(np.asarray(plane.transpose(*space_axes, channel_axis).data))

    task.progress(10, "Loading the model")
    predict, batch = _model()

    def on_tile(done: int, total: int) -> None:
        task.check_cancelled()
        task.progress(10 + 80 * done // total, f"Segmented {done} of {total} tiles")

    labels = segment_at_model_scale(rgb, predict, pixel_size_um, min_area_um2, batch=batch, on_tile=on_tile)

    task.progress(90, "Uploading labels")
    # The labels keep only the space axes: when the image had others (its channels, a
    # single timepoint), the edge is a projection that drops them.
    if tuple(space_axes) == tuple(str(d) for d in data.dims):
        edge = image.derive_identity(value_relation=ValueRelation.CATEGORIZED)
    else:
        edge = image.derive_projection(list(space_axes), value_relation=ValueRelation.CATEGORIZED)
    # No histogram anchor: contrast limits mean nothing for object ids.
    mask = mikro.create_array_dataset(
        data=xr.DataArray(labels, dims=space_axes),
        scales=[],
        name=f"HoverFast nuclei of {image.name}" if image.name else "HoverFast nuclei",
        axes=image.carried_axes(space_axes),
        anchors=[],
        derived_from=[edge],
    )
    if mask.intrinsic_system is None:
        raise RuntimeError(f"The uploaded labels {mask.id} have no intrinsic system")
    mask.intrinsic_system.stage(name="Default")

    task.progress(95, "Uploading measurements")
    table = mikro.create_table_dataset(
        name=f"{mask.name} measurements",
        data=nucleus_table(labels, pixel_size_um),
        columns=[
            # An INDEX axis identified by the mask: the table is what its ids are.
            ColumnInput(
                name="object_id",
                axis_type=AxisType.INDEX,
                long_name="nucleus id",
                identified_by=(DatasetIdentifiesInput(dataset=mask.id),),
            ),
            ColumnInput(name="area_px", role=ColumnRole.ATTRIBUTE, long_name="area (pixels)"),
            ColumnInput(name="area_um2", role=ColumnRole.ATTRIBUTE, long_name="area (µm²)"),
            ColumnInput(name="centroid_y", role=ColumnRole.ATTRIBUTE, long_name="centroid y (pixels)"),
            ColumnInput(name="centroid_x", role=ColumnRole.ATTRIBUTE, long_name="centroid x (pixels)"),
            ColumnInput(name="bbox_min_y", role=ColumnRole.ATTRIBUTE),
            ColumnInput(name="bbox_min_x", role=ColumnRole.ATTRIBUTE),
            ColumnInput(name="bbox_max_y", role=ColumnRole.ATTRIBUTE),
            ColumnInput(name="bbox_max_x", role=ColumnRole.ATTRIBUTE),
            ColumnInput(name="eccentricity", role=ColumnRole.ATTRIBUTE),
            ColumnInput(name="solidity", role=ColumnRole.ATTRIBUTE),
        ],
    )
    return mask.lens(), table.lens()


if __name__ == "__main__":
    run(app)
