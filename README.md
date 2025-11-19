# sfm-gsplat-pipeline
> Unified **Images / Video → SfM → Depth → 3D Gaussian Splatting** pipeline built on top of HLOC, FastMap, and GSplat.

This repository implements an end-to-end 3D reconstruction and rendering pipeline:

- Accepts raw **videos and/or image folders**
- Selects **sharp frames** via classical CV metrics
- Runs **Structure-from-Motion (SfM)** with HLOC and advanced backends (FastMap / GLoMap / COLMAP)
- Generates **COLMAP-aligned monocular depth maps**
- Trains a **3D Gaussian Splatting (GSplat)** model with optional depth guidance
- Produces ready-to-render **3D Gaussian splats** and a concise **SfM quality report**

It is designed for experimenting with SfM and 3DGS workflow.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Environment & Installation](#environment--installation)
5. [Quick Start](#quick-start)
6. [Datasets & Input Requirements](#datasets--input-requirements)
7. [Unified Pipeline CLI (`run_full_pipeline.py`) ](#unified-pipeline-cli-run_full_pipelinepy)
8. [SfM Only CLI (`scripts/run_hloc.py`)](#sfm-only-cli-scriptsrun_hlocpy)
9. [Depth Map Generation (`scripts/gen_aligned_depths.py`)](#depth-map-generation-scriptsgen_aligned_depthspy)
10. [Sharp Frame Selection (`scripts/sharp_frame_selector.py`)](#sharp-frame-selection-scriptssharp_frame_selectorpy)
11. [Output Layout](#output-layout)
12. [Presets & Configuration](#presets--configuration)
13. [Why This Project Is Special](#why-this-project-is-special)
14. [Acknowledgements](#acknowledgements)

---

## Overview

The pipeline:

1. (Optional) Extracts **sharp frames** from videos/image folders
2. Runs **SfM** with HLOC (and FastMap / GLoMap / COLMAP backends)
3. Validates the reconstruction and produces an **SfM quality report**
4. Generates **aligned depth maps** for all registered images
5. Trains a **GSplat** model (MCMC / standard) with optional depth-guided losses
6. Saves 3D Gaussian splats (PLY), training stats, and logs

---

## Features

- 🎬 **Raw Media → Dataset**
  - Sharp frame selection from videos or image folders using Laplacian, Sobel, Canny, or a combined score.

- 🏗 **Robust SfM**
  - HLOC front-end with many extractors/matchers and support for multiple SfM backends (FastMap / GLoMap / COLMAP).

- 📊 **SfM Quality Diagnostics**
  - `parse_colmap.py` reads COLMAP binaries and produces an SfM quality & 3DGS-readiness report.

- 🌊 **Depth Estimation**
  - `gen_aligned_depths.py` generates COLMAP-aligned depth maps using Depth-Anything-V2, ZoeDepth, or MiDaS.

- 🌐 **3D Gaussian Splatting**
  - Integrated GSplat training (including MCMC strategy) with options for depth-guided training.

- 🎚 **Presets & Configs**
  - YAML presets (`fast`, `balanced`, `quality`, `ultra`, etc.) that jointly tune SfM & GSplat, overridable from CLI.

- 🧪 **Metrics & Logging**
  - Rich console logs and sharpness statistics for better debugging and dataset triage.

- 🧩 **Modular Entry Points**
  - One-command full pipeline, plus separate CLIs for SfM, depth, and sharp-frame selection.

---

## Project Structure

Simplified layout:

```text
sfm-gsplat-pipeline/
├── README.md                # This file
├── requirements.txt         # Python dependencies for the core pipeline
├── run_pipeline.py          # Thin wrapper for the unified pipeline
├── scripts/                 # CLI tools for each stage (legacy / direct use)
│   ├── run_full_pipeline.py       # Original unified pipeline entrypoint
│   ├── run_hloc.py               # Standalone HLOC + SfM runner
│   ├── gen_aligned_depths.py     # COLMAP-aligned depth generation
│   ├── sharp_frame_selector.py   # Sharp frame selection from videos/images
│   ├── parse_colmap.py, auto_3dgs.py, ...  # utilities & helpers
├── configs/                 # YAML configs / presets (fast, balanced, quality, ultra, ...)
├── src/
│   ├── sfm_gsplat_pipeline/      # Your package: pipeline, presets, utils
│   │   ├── cli.py                # Modern CLI entrypoint used by run_pipeline.py
│   │   ├── pipeline.py           # Pipeline wrapper (re-exporting the main Pipeline class)
│   │   ├── presets.py            # Built-in presets (fast/balanced/quality/ultra...)
│   │   └── utils/                # Config, IO, sharpness, depth, and command helpers
│   ├── hloc/                     # Modified HLOC fork integrated with this pipeline
│   ├── fastmap/                  # Modified FastMap fork integrated with this pipeline
│   └── gsplat/                   # Modified GSplat fork integrated with this pipeline
└── third_party/
    ├── LightGlue, DKM, DeDoDe, Dust3R, Mast3r, Depth-Anything-V2, ...
    └── sharp-frame-extractor/    # Original sharp frame extractor reference
```

**Note on `src/` vs `third_party/`:**

- `src/hloc`, `src/fastmap`, and `src/gsplat` are **locally modified forks** of the upstream projects, adapted specifically for this pipeline.
- Code under `third_party/` is generally kept closer to upstream and used more selectively (e.g. for additional feature extractors, matchers, or depth models).

High-level flow:

- **Raw media** → `sharp_frame_selector.py` → `data/<scene>/images/`
- **SfM** → `run_hloc.py` → `data/<scene>/sparse/`
- **Depth** → `gen_aligned_depths.py` → `data/<scene>/depths/`
- **3DGS** → GSplat `simple_trainer` → `data/<scene>/results/`

---

## Environment & Installation

The project targets a **GPU-accelerated PyTorch** environment (CUDA). CPU will work but is much slower.

### 1. Clone Repository (and submodules, if used)

```bash
git clone <your-repo-url>.git
cd sfm-gsplat-pipeline

# If you are using git submodules for third_party:
git submodule update --init --recursive
```

### 2. Create Environment & Install Dependencies

Using conda:

```bash
conda create -n sfm-gsplat python=3.11 -y
conda activate sfm-gsplat

pip install -r requirements.txt
```

Then install local packages in editable mode:

```bash
pip install -e ./src/hloc
pip install -e ./src/fastmap
pip install -e ./src/gsplat
```

### 3. Install GPU PyTorch

Follow official PyTorch instructions for your CUDA version, for example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Depth-Anything-V2 / ZoeDepth / MiDaS

- For Depth-Anything-V2, place checkpoints in `third_party/Depth-Anything-V2/checkpoints/` as documented in that repo.
- ZoeDepth and MiDaS are loaded via `torch.hub` when selected.

---

## Quick Start

### A) Full Pipeline: Raw Media → 3DGS

```bash
cd scripts

python run_full_pipeline.py     --data_dir ../data/garden     --media /path/to/video_or_image_folder     --preset balanced     --with-depth     --depth-model depth_anything_v2     --depth-device cuda
```

This will:

1. Extract sharp frames into `data/garden/images/` (if images are not already present).
2. Run HLOC-based SfM with the configured backend.
3. Produce an SfM quality & 3DGS readiness report.
4. Generate aligned depth maps into `data/garden/depths/` (when `--with-depth` is set).
5. Launch GSplat training and write outputs to `data/garden/results/`.

### B) SfM Only

```bash
cd scripts

python run_hloc.py     --project_path ../data/garden     --extractor disk     --matcher disk-lightglue     --global_extractor netvlad     --sfm-backend fastmap
```

This produces a COLMAP sparse model in `data/garden/sparse/`.

---

## Datasets & Input Requirements

Recommended:

- 50–500 sharp frames per scene
- 60–80% overlap between neighboring views
- Low motion blur (use sharp-frame selector on videos)
- 1–4K resolution images
- Good coverage around the scene/object

Inputs:

- Direct images in `<data_dir>/images/`
- Or raw media passed via `--media` (videos and/or image folders)

---

## Unified Pipeline CLI (`run_full_pipeline.py`)

Main entrypoint: `scripts/run_full_pipeline.py`.

### Basic Usage

```bash
python scripts/run_full_pipeline.py     --data_dir ./data/garden     --preset balanced     --with-depth
```

### Key Arguments (selection)

**Core:**

- `--data_dir`: Root data directory for a scene (e.g. `./data/garden`)
- `--result_dir`: Optional explicit output directory for GSplat results

**Presets / Config:**

- `--preset`: One of `fast`, `balanced`, `quality`, `ultra`, `custom_expert`, etc.
- `--config`: Specific YAML config path or name inside `configs/`
- `--auto-preset`: If enabled, auto-selects a preset based on dataset properties

**Depth:**

- `--with-depth`: Enable depth map generation & depth-guided training
- `--depth-model`: `depth_anything_v2`, `zoedepth`, or `midas`
- `--depth-device`: `cuda` or `cpu`
- `--depth-dir`: Custom output for depth maps (default: `<data_dir>/depths`)

**Sharp Frame Extraction:**

- `--media`: One or more raw media inputs (videos or image folders)
- `--sharp-method`: `laplacian`, `sobel`, `canny`, or `combined`
- `--sharp-top-k`: Maximum number of frames per input (0 = all above threshold)
- `--sharp-min-score`: Minimum sharpness score
- `--sharp-frame-step`: Analyze every N-th frame in videos

**Advanced Overrides:**

You can override config keys from the CLI, for example:

```bash
python scripts/run_full_pipeline.py     --data_dir ./data/garden     --config balanced     --hloc-override num_matched=32     --gs-override batch_size=2     --gs-override depth_lambda=0.05
```

---

## SfM Only CLI (`scripts/run_hloc.py`)

Wrapper around HLOC with support for multiple local/global features and matchers.

Example:

```bash
python scripts/run_hloc.py     --project_path ./data/garden     --extractor superpoint_aachen     --matcher superglue     --global_extractor netvlad     --matching-method retrieval     --num_matched 20     --sfm-backend fastmap
```

Highlights:

- Discovers available extractors, matchers, and global models
- Supports retrieval / sequential / exhaustive matching
- Can be followed by parsing & reporting via `parse_colmap.py`

---

## Depth Map Generation (`scripts/gen_aligned_depths.py`)

Generates COLMAP-aligned depth maps for each registered image.

```bash
python scripts/gen_aligned_depths.py     --colmap_sparse_path ./data/garden/sparse/0     --image_dir ./data/garden/images     --output_dir ./data/garden/depths     --model_type depth_anything_v2     --device cuda
```

Key options:

- `--model_type`: `depth_anything_v2`, `depth_anything_v2_hf`, `zoedepth`, `midas`
- `--depth_dtype`: `float16`, `float32`, `float64`
- `--no_compress`: store `.npy` instead of `.npz`

---

## Sharp Frame Selection (`scripts/sharp_frame_selector.py`)

Selects sharp frames from videos or image folders and logs quality metrics.

```bash
python scripts/sharp_frame_selector.py     /path/to/video1.mp4 /path/to/video2.mp4     /path/to/image_folder     -o ./data/garden/images     --mode auto     --method combined     --frame-step 2     --top-k 300
```

What it does:

- Supports multiple inputs (videos + image folders)
- Computes Laplacian, Sobel, Canny, and combined scores
- Keeps only the sharpest frames
- Can be used as a standalone pre-processing tool

---

## Output Layout

Typical outputs for `data/garden/` after a full run:

```text
data/garden/
├── images/              # Selected sharp frames (or original images)
├── images_4/, images_8/ # Downsampled variants for GSplat
├── hloc_outputs/        # HLOC features, matches, logs
├── sparse/
│   └── 0/               # COLMAP sparse model
├── depths/              # Aligned depth maps
└── results/
    ├── ckpts/           # GSplat checkpoints
    ├── stats/           # JSON metrics, training logs
    ├── renders/         # Rendered views (if configured)
    └── *.ply            # Exported Gaussian splats
```

---

---

## Interactive 3DGS Viewer (GSplat)

After training, you can inspect the 3D Gaussian splats interactively using the GSplat simple viewer (based on `viser`).

Example (for the `jeep` scene):

```bash
python src/gsplat/examples/simple_viewer.py \
    --ckpt ./data/jeep/results/ckpts/ckpt_29999_rank0.pt \
    --output_dir ./data/jeep/results \
    --scene_root ./data/jeep \
    --colmap_sparse sparse/0 \
    --images_subdir images \
    --port 8082
```

Then open the displayed URL (e.g. `http://localhost:8082`) in your browser to orbit, zoom, and inspect the reconstruction.

The same pattern works for any scene:

```bash
python src/gsplat/examples/simple_viewer.py \
    --ckpt ./data/<scene>/results/ckpts/ckpt_29999_rank0.pt \
    --output_dir ./data/<scene>/results \
    --scene_root ./data/<scene> \
    --colmap_sparse sparse/0 \
    --images_subdir images \
    --port 8080
```

## Presets & Configuration

The `configs/` folder and the `PRESETS` dictionary in `run_full_pipeline.py` encode:

- SfM settings (extractor, matcher, num_matched, backend, camera model, data_factor)
- GSplat settings (subcommand, max steps, eval steps, SH strategy, batch size, depth lambda, etc.)

Typical presets:

- `fast` – quick preview
- `balanced` – default quality/speed trade-off
- `quality` / `ultra` – higher quality settings
- `custom_expert` – starting point for manual tuning

You can edit YAML files directly or override individual values via the CLI.

---

## Why This Project Is Special

From an engineering perspective, this project shows:

1. **End-to-end system design**  
   - A full chain from raw media to 3D Gaussian splats, not just isolated models.

2. **Deep integration of modern research code**  
   - HLOC, FastMap / GLoMap / COLMAP, state-of-the-art monocular depth, and GSplat working together.

3. **Practical considerations**  
   - Sharp-frame pre-filtering to protect SfM and 3DGS from poor inputs.
   - Presets for different compute budgets.
   - Clear logging, metrics, and a standard folder layout.

4. **Portfolio-friendly clarity**  
   - Clean CLIs, structured configs, and this README make the project easy to understand how SfM and 3DGS work.

---

## Acknowledgements

This project builds heavily on the work of the following projects and their authors:

- **AutoHLOC** – for inspiration and one-click HLOC workflows  
  - https://github.com/AIG3DX/AutoHLOC
- **HLOC** – Hierarchical Localization framework used as the SfM front-end  
  - https://github.com/cvg/hloc
- **FastMap** – SfM backend integrated into this pipeline  
  - https://github.com/pals-ttic/fastmap
- **GSplat** – 3D Gaussian Splatting library used for training and rendering  
  - https://github.com/nerfstudio-project/gsplat

Additional third-party modules under `third_party/` (e.g. LightGlue, DKM, DeDoDe, Dust3R, Mast3r, Depth-Anything-V2, and others) are used under their respective licenses. Please refer to the original repositories for license details and academic citations.
