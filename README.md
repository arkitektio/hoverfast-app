# hoverfast-app

[HoverFast](https://github.com/choosehappy/HoverFast) nuclei segmentation of H&E images, as an [Arkitekt](https://arkitekt.live) app.

HoverFast is a small multi-scale U-Net, distilled from HoVer-Net. It predicts where nuclei are and, for every pixel, how far it is from the centre of its nucleus. A watershed then splits touching nuclei along those maps. This app runs HoverFast's pretrained cross-tissue model on an RGB image in mikro and returns two things:

- the nuclei as a **label mask** on the image's own grid, with each nucleus under its own id;
- a **table** with one row per nucleus, keyed by its id in the mask: area (pixels and µm²), centroid, bounding box, eccentricity and solidity.

## The action

**Segment Nuclei (HoverFast)** (`segment_nuclei`)

| Parameter | Default | |
|---|---|---|
| `image` | | An RGB image: two space axes and one channel axis of three, with at most one timepoint |
| `pixel_size_um` | `0.25` | How wide a pixel of the image is, in µm |
| `min_area_um2` | `5.0` | Nuclei smaller than this are left out |

The model was trained at **0.25 µm/px (40x)**. For any other `pixel_size_um` (off by more than 10%), the image is resampled to 0.25 µm/px for the model, and the labels are resampled back. For a 20x image, pass `0.5`.

Large images are segmented in tiles of 1024 pixels that overlap by 128, as HoverFast's whole-slide pipeline does. Each nucleus is kept by the one tile whose core holds its centroid. One tile at a time is predicted on a CPU and four on a GPU.

## Flavours

| Flavour | torch | Platforms |
|---|---|---|
| `vanilla` | CPU (fp32) | linux/amd64, linux/arm64 |
| `cuda` | CUDA 12.6 (fp16); needs an NVIDIA driver ≥ 560 | linux/amd64 |

## Development

HoverFast is not on PyPI. It is installed from its git repository, pinned to a commit of its `dev` branch (see `pyproject.toml`). torch comes from PyTorch's own index, so always pick a build with an extra:

```bash
uv sync --extra cpu            # or --extra cu126 on an NVIDIA GPU
uv run --extra cpu pytest      # the tests without a server
uv run --extra cpu pytest -m hub   # against a hub made for the tests (Docker + konstruktor)
arkitekt run dev
```

The code is laid out as follows:

- `app.py`: the app and its action.
- `src/hoverfast_app/model.py`: loads the weights and predicts on tiles in memory, in fp16 on a GPU and fp32 on a CPU.
- `src/hoverfast_app/segment.py`: the tiling and HoverFast's own watershed (`pre_watershed`, `watershed_object_roi`), which turn the model's output into labels without writing files.
- `src/hoverfast_app/measure.py`: the per-nucleus table.

The pretrained weights (`src/hoverfast_app/weights/hoverfast_crosstissue_best_model.safetensors`) are HoverFast's own, taken from the same commit. Their source and sha256 are recorded in `model.py`.

## Credit

HoverFast is by Petros Liakopoulos, Julien Massonnet, Andrew Janowczyk and co-authors. It is BSD-3-Clause, Copyright (c) 2024, Andrew Janowczyk, and so are its weights that ship with this app. If you use it, cite their paper, as HoverFast's README asks:

```bibtex
@article{Liakopoulos2024, doi = {10.21105/joss.07022}, url = {https://doi.org/10.21105/joss.07022}, year = {2024}, publisher = {The Open Journal}, volume = {9}, number = {101}, pages = {7022}, author = {Petros Liakopoulos and Julien Massonnet and Jonatan Bonjour and Medya Tekes Mizrakli and Simon Graham and Michel A. Cuendet and Amanda H. Seipel and Olivier Michielin and Doron Merkler and Andrew Janowczyk}, title = {HoverFast: an accurate, high-throughput, clinically deployable nuclear segmentation tool for brightfield digital pathology images}, journal = {Journal of Open Source Software} }
```
