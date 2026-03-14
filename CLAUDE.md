# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video2Depth generates depth maps from videos and images using pluggable depth estimation models. It supports multiple models (MiDaS, Depth Anything V2, ZoeDepth, Marigold) via a registry-based architecture.

## Commands

```bash
# Install dependencies
pip install torch torchvision opencv-python timm Pillow numpy ffmpeg-python

# Run on video (default model: midas_large)
python3 video2depth.py --video [filename]

# Run on image
python3 video2depth.py --image [filename]

# Specify model
python3 video2depth.py -v [filename] -m depth_anything_v2
```

There is no test suite or build system. FFmpeg must be installed on the system.

## Architecture

### Pipeline (`video2depth.py`)

Entry point with a linear processing pipeline:

1. **`video2image()`** — Extracts video frames to `out/<name>/image/` using FFmpeg
2. **`image2depth(output_dir, model)`** — Runs depth estimation on each frame using a `DepthModel` instance, outputs to `out/<name>/depth/`
3. **`merge_image_depth()`** — Creates vertically-stacked original+depth composites in `out/<name>/merged/`
4. **`depth2video()`** — Reassembles frames into MP4 videos (with and without original audio)

The script has resume capability — it skips steps where output files already exist by comparing file counts.

### Model System (`models/`)

- **`base.py`** — `DepthModel` ABC with `load()`, `predict(img_rgb) -> uint8 depth map`, `name` property
- **`__init__.py`** — Lazy model registry using `importlib` (dependencies only loaded when model is selected)
- **`midas.py`**, **`depth_anything_v2.py`**, **`zoedepth.py`**, **`marigold.py`** — Model implementations

### Adding a New Model

1. Create `models/<name>.py` with a class extending `DepthModel`
2. Implement `load()`, `predict()`, `name`
3. Add entry to `MODEL_REGISTRY` in `models/__init__.py`

## Key Details

- Default model: `midas_large` (backward compatible)
- GPU (CUDA) auto-detected; falls back to CPU
- Output goes to `out/<input_filename>/` with subdirs `image/`, `depth/`, `merged/`
- Marigold requires extra deps: `diffusers>=0.25 accelerate`
