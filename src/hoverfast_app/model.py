"""The pretrained HoverFast model, and a prediction on tiles held in memory.

The model is HoverFast's own (``hoverfast.models``), loaded by its own loader. Only
the prediction is written here: upstream's ``roi.pipeline.predict_roi`` always casts
its input to fp16 and keeps the autograd graph, which a CPU cannot run well. This
one runs in the model's own precision, fp16 on a GPU and fp32 on a CPU.
"""

from pathlib import Path

import numpy as np
import torch
from hoverfast.models.trt_engine import load_eager_model

# The weights HoverFast publishes (BSD-3, Copyright (c) 2024, Andrew Janowczyk),
# from the commit the `hoverfast` dependency is pinned to. Trained at 0.25 µm/px (40x).
WEIGHTS_URL = (
    "https://raw.githubusercontent.com/choosehappy/HoverFast/"
    "66c97aa4893f0f823cef17eec4a45dc7009356eb/hoverfast_crosstissue_best_model.safetensors"
)
WEIGHTS_SHA256 = "c41bdaa655c12533754701d32942768e837e4e9b4c6bca58c137b44b1dc39bb7"
DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "hoverfast_crosstissue_best_model.safetensors"

# What upstream mirrors around a region before it predicts, against the border effects.
_CONTEXT = 84


def load(path: Path = DEFAULT_WEIGHTS, device: torch.device | None = None) -> torch.nn.Module:
    """The model, on the GPU when there is one: fp16 there, as upstream runs it, and fp32 on a CPU."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_eager_model(str(path), device)
    return model if device.type == "cuda" else model.float()


def predict(regions: np.ndarray, model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray]:
    """Where the nuclei are in (N, H, W, 3) uint8 RGB regions, and their distance maps.

    Returns a boolean (N, H, W) nucleus mask and the float32 (N, 2, H, W) horizontal
    and vertical maps that `hoverfast.wsi.postprocess.pre_watershed` splits touching
    nuclei with. Regions of any size from 85 pixels up are taken: they are mirrored
    by 84 pixels, and further to a multiple of 4, which the model needs.
    """
    _, height, width, _ = regions.shape
    bottom = _CONTEXT + (-(height + 2 * _CONTEXT)) % 4
    right = _CONTEXT + (-(width + 2 * _CONTEXT)) % 4
    padded = np.pad(regions, ((0, 0), (_CONTEXT, bottom), (_CONTEXT, right), (0, 0)), mode="reflect")

    parameter = next(model.parameters())
    with torch.inference_mode():
        batch = torch.from_numpy(padded).to(parameter.device).permute(0, 3, 1, 2)
        batch = batch.to(parameter.dtype).div_(255).contiguous(memory_format=torch.channels_last)
        output, maps = model(batch)
        window = (slice(None), slice(None), slice(_CONTEXT, _CONTEXT + height), slice(_CONTEXT, _CONTEXT + width))
        mask = output[window].argmax(dim=1).bool().cpu().numpy()
        maps = maps[window].float().cpu().numpy()
    return mask, maps
