#!/usr/bin/env python3
"""
AutoHLOC + GSplat: Unified End-to-End 3D Gaussian Splatting Pipeline
One script to rule them all: from raw images to trained 3DGS

Usage (examples):

  # Auto mode (analyzes dataset, picks preset)
  python run_full_pipeline.py --data_dir ./data/garden

  # Manual built-in preset
  python run_full_pipeline.py --data_dir ./data/garden --preset balanced

  # External config (overrides built-in presets)
  python run_full_pipeline.py --data_dir ./data/garden --config configs/balanced.yaml

  # Config + symbolic overrides (no quotes):
  python run_full_pipeline.py \
      --data_dir ./data/garden \
      --config balanced \
      --hloc-override matching_method=sequential \
      --hloc-override num_matched=32 \
      --gs-override batch_size=2 \
      --gs-override strategy.cap_max=2000000 \
      --gs-override depth_lambda=0.1

  # Emergency raw pass-through:
  python run_full_pipeline.py \
      --data_dir ./data/garden \
      --hloc-flags "--vis --matching-method sequential" \
      --gs-flags "--pose-opt --antialiased"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Tuple, Dict, Any
from rich.console import Console
from rich.table import Table
import logging
import shlex
import ast

# Optional: for auto-preset detection
try:
    from PIL import Image
    import numpy as np
    AUTO_PRESET_AVAILABLE = True
except ImportError:
    AUTO_PRESET_AVAILABLE = False

# Optional: YAML config for external presets
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

# Setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
console = Console()

# =============================================================================
# PRESET CONFIGURATIONS (built-in; can be overridden by --config YAML)
# =============================================================================


# Built-in preset names used for CLI choices and simple defaults.
# The actual detailed configuration can be provided via YAML files in `configs/`.
# When a YAML config is present (via --config or --preset treated as a config
# name), it takes precedence. These built-ins mainly provide human-readable
# descriptions and fallback behaviour.
PRESETS: dict[str, dict[str, object]] = {
    "fast": {
        "description": "Fast preset (lower quality, shorter runtime)",
        "expected_time": "5–10 min",
    },
    "balanced": {
        "description": "Balanced quality/speed preset",
        "expected_time": "15–25 min",
    },
    "quality": {
        "description": "Higher quality preset (longer training)",
        "expected_time": "25–45 min",
    },
    "ultra": {
        "description": "Max quality preset (long training; high VRAM)",
        "expected_time": "45–90 min",
    },
    # Benchmark / expert configs – typically backed by YAML files under configs/.
    "benchmark_disk_lightglue": {
        "description": "Benchmark preset: DISK + LightGlue under fixed GSplat recipe",
        "expected_time": "20–35 min",
    },
    "benchmark_xfeat_lighterglue": {
        "description": "Benchmark preset: XFeat + LighterGlue",
        "expected_time": "20–35 min",
    },
    "benchmark_template": {
        "description": "Template benchmark preset for custom experiments",
        "expected_time": "unknown",
    },
    "custom_expert": {
        "description": "Expert custom preset (expects external YAML overrides)",
        "expected_time": "unknown",
    },
    "custom_preview": {
        "description": "Preview preset for quick inspection",
        "expected_time": "5–15 min",
    },
}

def analyze_dataset_and_recommend_preset(images_dir: Path) -> tuple[str, dict[str, object]]:
    """Very lightweight heuristic to suggest a preset based on image count & resolution.

    This does not have to be perfect – it just helps choose between fast/balanced/quality.
    """
    info: dict[str, object] = {}
    if not AUTO_PRESET_AVAILABLE or not images_dir.exists():
        info["num_images"] = 0
        info["megapixels"] = 0.0
        info["reason"] = "Auto-preset disabled or images directory not found; defaulting to 'balanced'."
        return "balanced", info

    img_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    num_images = len(img_paths)
    info["num_images"] = num_images

    if num_images == 0:
        info["megapixels"] = 0.0
        info["reason"] = "No images found; defaulting to 'balanced'."
        return "balanced", info

    # Sample up to 10 images to estimate average megapixels
    sample_paths = img_paths[:10]
    total_mp = 0.0
    for p in sample_paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
            total_mp += (w * h) / 1e6
        except Exception:
            continue
    avg_mp = total_mp / max(1, len(sample_paths))
    info["megapixels"] = avg_mp

    # Simple heuristic: smaller / fewer images -> fast; medium -> balanced; large -> quality.
    if num_images <= 50 or avg_mp <= 1.0:
        preset = "fast"
        reason = "Small dataset or low resolution; using fast preset."
    elif num_images <= 250 and avg_mp <= 6.0:
        preset = "balanced"
        reason = "Moderate dataset size/resolution; using balanced preset."
    else:
        preset = "quality"
        reason = "Large or high-res dataset; using quality preset."

    info["reason"] = reason
    return preset, info


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply symbolic overrides like ['strategy.cap_max=2000000', 'batch_size=2'].

    Supports dotted keys to update nested dictionaries.
    """
    if not overrides:
        return
    for item in overrides:
        if not item:
            continue
        if "=" not in item:
            continue
        key_str, value_str = item.split("=", 1)
        key_str = key_str.strip()
        value_str = value_str.strip()

        # Try to interpret value as Python literal, otherwise keep as string
        try:
            value = ast.literal_eval(value_str)
        except Exception:
            value = value_str

        # Support dotted keys for nested dicts (e.g. strategy.cap_max)
        parts = key_str.split(".")
        d = cfg
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value


def extend_cmd_from_config(cmd: list[str], prefix: str, cfg: dict) -> None:
    """Map a flat config dict into CLI flags.

    Example:
      cfg = {"max_steps": 30000, "antialiased": True}
      ->  --max-steps 30000 --antialiased

    If *prefix* is provided (e.g. "strategy."), keys are prefixed before turning
    dots/underscores into CLI-style flags. For GSplat's strategy options, the CLI
    expects `--strategy.<option>` (with a dot), e.g. `--strategy.cap-max`.
    """
    if not cfg:
        return

    for key, value in cfg.items():
        if value is None:
            continue

        # GSplat strategy options: keep the 'strategy.' prefix as a dot group, and
        # only dashify the actual key part.
        if prefix == "strategy.":
            flag = "--strategy." + key.replace("_", "-")
        else:
            full_key = f"{prefix}{key}" if prefix else key
            flag = "--" + full_key.replace(".", "-").replace("_", "-")

        # Booleans -> flag only when True
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue

        # List/tuple -> repeat flag
        if isinstance(value, (list, tuple)):
            for item in value:
                cmd.extend([flag, str(item)])
            continue

        # Scalar -> flag + value
        cmd.extend([flag, str(value)])





# =============================================================================
# HELPERS
# =============================================================================

class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.data_dir = Path(args.data_dir).resolve()
        self.images_dir = self.data_dir / "images"
        self.sparse_dir = self.data_dir / "sparse" / "0"
        self.result_dir = Path(args.result_dir) if args.result_dir else self.data_dir / "results"

        # Depth output directory (for depth-guided training)
        if args.depth_dir:
            self.depth_dir = Path(args.depth_dir).resolve()
        else:
            self.depth_dir = self.data_dir / "depths"
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        # pass-through flags (emergency escape hatch)
        self.hloc_flags = shlex.split(args.hloc_flags) if args.hloc_flags else []
        self.gs_flags = shlex.split(args.gs_flags) if args.gs_flags else []

        # Raw media inputs (videos and/or image folders) for sharp-frame selection
        self.media_inputs = [Path(p).resolve() for p in (args.media or [])]
        self.sharp_selector_script = Path(args.sharp_selector_script).resolve()
        self.sharp_method = args.sharp_method
        self.sharp_top_k = args.sharp_top_k
        self.sharp_min_score = args.sharp_min_score
        self.sharp_frame_step = args.sharp_frame_step
        self.sharp_verbose = args.sharp_verbose
        self.sharp_temporal_method = args.sharp_temporal_method
        self.sharp_save_metrics_json = args.sharp_save_metrics_json
        self.sharp_save_temporal_plot = args.sharp_save_temporal_plot

        # Viser viewer integration (optional)
        self.with_viewer = getattr(args, "with_viewer", False)
        self.viewer_port = getattr(args, "viewer_port", 8080)
        self.viewer_poll_seconds = getattr(args, "viewer_poll_seconds", 5.0)
        self.viewer_process: subprocess.Popen | None = None

        # Optional GSplat simple_viewer integration (post-training or existing ckpt)
        self.with_gsplat_viewer = getattr(args, "with_gsplat_viewer", False)
        self.gs_viewer_port = getattr(args, "gs_viewer_port", 8081)
        self.gs_viewer_process: subprocess.Popen | None = None

        # Determine preset (built-in or YAML)
        if args.config:
            if yaml is None:
                raise RuntimeError("PyYAML is required for --config. Install: pip install pyyaml")

            cfg_path = Path(args.config)
            if not cfg_path.exists():
                # allow shorthand: --config balanced -> configs/balanced.yaml
                if not cfg_path.suffix:
                    cand = Path("configs") / f"{cfg_path.name}.yaml"
                else:
                    cand = Path("configs") / cfg_path.name
                if cand.exists():
                    cfg_path = cand

            preset_cfg = yaml.safe_load(cfg_path.read_text())
            preset_cfg.setdefault("description", f"External config: {cfg_path.name}")
            preset_cfg.setdefault("expected_time", "unknown")
            self.preset_name = preset_cfg.get("name", "custom")
            self.preset = preset_cfg
        elif args.auto_preset:
            # For auto-preset we still expect images_dir to exist (or be created by media preprocessing)
            if not self.images_dir.exists() and not self.media_inputs:
                raise FileNotFoundError(f"Images directory not found for auto-preset: {self.images_dir}")
            recommended_preset, info = analyze_dataset_and_recommend_preset(self.images_dir)
            console.print("\n📊 Dataset Analysis:", style="bold cyan")
            console.print(f"   • Number of images: {info.get('num_images', 'N/A')}", style="cyan")
            console.print(f"   • Average resolution: {info.get('megapixels', 'N/A'):.1f} MP", style="cyan")
            console.print(f"   • Recommended preset: {recommended_preset}", style="bold green")
            console.print(f"   • Reason: {info['reason']}", style="green")
            console.print()
            self.preset_name = recommended_preset
            self.preset = PRESETS[self.preset_name]
        else:
            self.preset_name = args.preset
            self.preset = PRESETS[self.preset_name]

        # Ensure sfm / gsplat dicts exist
        self.preset.setdefault("sfm", {})
        self.preset.setdefault("gsplat", {})

        # Apply symbolic overrides from CLI (hloc-override / gs-override)
        apply_overrides(self.preset["sfm"], args.hloc_override or [])
        apply_overrides(self.preset["gsplat"], args.gs_override or [])

        # Validate / ensure images_dir will be available
        if not self.images_dir.exists() and not self.media_inputs:
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir} and no --media inputs were provided."
            )

        # If images_dir already exists, make sure parent dirs are in place
        if self.images_dir.exists():
            self.images_dir.mkdir(parents=True, exist_ok=True)

        # Create output dirs
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # Stage ordering & numbering (depends on flags)
        self.stage_order = []
        if self.media_inputs:
            self.stage_order.append("sharp")
        self.stage_order.append("sfm")
        if self.args.with_depth:
            self.stage_order.append("depth")
        self.stage_order.append("training")
        self.total_stages = len(self.stage_order)
        self.stage_indices: Dict[str, int] = {
            name: idx + 1 for idx, name in enumerate(self.stage_order)
        }

        # Timing
        self.timings: Dict[str, float] = {}
        self.start_time: float | None = None

    # -------------------------------------------------------------------------

    def _find_latest_ckpt(self):
        """Return the most recent GSplat checkpoint in result_dir/ckpts or result_dir, or None."""
        ckpt_dir = self.result_dir / "ckpts"
        candidates = []
        if ckpt_dir.exists():
            candidates.extend(ckpt_dir.glob("*.pt"))

        # Fallback: look directly under result_dir
        if not candidates:
            candidates.extend(self.result_dir.glob("ckpt_*.pt"))
            candidates.extend(self.result_dir.glob("*.pt"))

        if not candidates:
            return None

        try:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
        except Exception:
            return None
        return latest

    # -------------------------------------------------------------------------

    def _stage_header(self, stage_name: str, title: str, icon: str) -> None:
        """Print a nice stage header with dynamic numbering."""
        idx = self.stage_indices.get(stage_name)
        total = self.total_stages
        console.print("\n" + "─" * 80, style="cyan")
        if idx is not None:
            console.print(f"{icon} STAGE {idx}/{total}: {title}", style="bold cyan")
        else:
            console.print(f"{icon} {title}", style="bold cyan")
        console.print("─" * 80 + "\n", style="cyan")

    # -------------------------------------------------------------------------

    def start_gsplat_viewer(self, ckpt_path: Path) -> None:
        """Optionally launch GSplat's simple_viewer on a given checkpoint."""
        if not getattr(self, "with_gsplat_viewer", False):
            return

        gsplat_dir = (Path(__file__).parent.parent / "src" / "gsplat").resolve()
        viewer_script = gsplat_dir / "examples" / "simple_viewer.py"
        if not viewer_script.exists():
            console.print(
                f"[yellow]⚠️ GSplat simple_viewer not found at {viewer_script}; skipping --with-gsplat-viewer.[/yellow]"
            )
            return

        cmd = [
            sys.executable,
            str(viewer_script),
            "--ckpt",
            str(ckpt_path),
            "--output_dir",
            str(self.result_dir),
            "--port",
            str(self.gs_viewer_port),
        ]

        console.print(f"🪟 Launching GSplat simple_viewer: {' '.join(cmd)}", style="dim")
        try:
            env = os.environ.copy()
            if self.args.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            self.gs_viewer_process = subprocess.Popen(cmd, cwd=str(gsplat_dir), env=env)
        except Exception as exc:
            console.print(
                f"[yellow]⚠️ Failed to launch GSplat simple_viewer: {exc}[/yellow]"
            )
            self.gs_viewer_process = None

    # -------------------------------------------------------------------------

    def start_viewer(self) -> None:
        """Optionally launch the Viser SfM+GSplat viewer in parallel."""
        if not getattr(self, "with_viewer", False):
            return

        viewer_script = Path(__file__).resolve().parent / "viser_sfm_gsplat_viewer_pro.py"
        if not viewer_script.exists():
            console.print(
                f"[yellow]⚠️ Viewer script not found at {viewer_script}; skipping --with-viewer.[/yellow]"
            )
            return

        cmd = [
            sys.executable,
            str(viewer_script),
            "--scene-root",
            str(self.data_dir),
            "--colmap-sparse",
            "sparse/0",
            "--images-subdir",
            "images",
            "--ply-dir",
            str(self.result_dir),
            "--poll-seconds",
            str(self.viewer_poll_seconds),
            "--port",
            str(self.viewer_port),
        ]

        console.print(f"🪟 Launching Viser viewer: {' '.join(cmd)}", style="dim")
        try:
            self.viewer_process = subprocess.Popen(cmd)
        except Exception as exc:
            console.print(
                f"[yellow]⚠️ Failed to launch viewer: {exc}[/yellow]"
            )
            self.viewer_process = None

    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Execute full pipeline."""
        console.print("\n" + "=" * 80, style="bold cyan")
        console.print("🚀 Unified 3DGS Pipeline: Images → SfM → Depth → Training", style="bold cyan")
        console.print("=" * 80 + "\n", style="bold cyan")

        console.print(f"📂 Data directory: {self.data_dir}", style="cyan")
        console.print(f"🎯 Preset: {self.preset_name} ({self.preset.get('description', '')})", style="cyan")
        console.print(f"⏱  Expected time: {self.preset.get('expected_time', 'unknown')}", style="cyan")
        console.print(f"🌊 Depth enabled: {self.args.with_depth}", style="cyan")
        console.print()

        self.start_time = time.time()

        # Optionally start Viser viewer
        self.start_viewer()

        try:
            # Optional Stage: sharp frame selection from raw media
            if self.media_inputs:
                self.run_sharp_frame_selection()

            # Stage: SfM
            if not self.args.skip_sfm:
                self.run_sfm()
            else:
                # still print stage header with numbering
                self._stage_header("sfm", "Structure from Motion (SfM)", "📷")
                console.print("⏭️  Skipping SfM (using existing)", style="yellow")

            # Validation (no stage number; avoid double parse_colmap)
            self.validate_reconstruction()

            # Optional Stage: depth
            if self.args.with_depth:
                self.run_depth()
            else:
                console.print(
                    "⏭️  Depth maps stage skipped (run with --with-depth to enable).",
                    style="yellow",
                )

            # Stage: training
            if not self.args.skip_training:
                self.run_gsplat()
            else:
                self._stage_header("training", "Training 3D Gaussian Splatting", "🎨")
                console.print("⏭️  Skipping training", style="yellow")
                # If requested, still try to launch GSplat viewer on any existing checkpoint
                if self.with_gsplat_viewer and self.gs_viewer_process is None:
                    ckpt_path = self._find_latest_ckpt()
                    if ckpt_path is None:
                        console.print(
                            f"[yellow]⚠️ --with-gsplat-viewer was requested, but no checkpoint was found in {self.result_dir}.[/yellow]"
                        )
                    else:
                        self.start_gsplat_viewer(ckpt_path)

            self.print_summary()

        except Exception as e:
            console.print(f"\n❌ Pipeline failed: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # -------------------------------------------------------------------------

    def run_sharp_frame_selection(self) -> None:
        """Stage: extract sharp frames from raw media into images/."""
        self._stage_header("sharp", "Sharp frame selection from media", "🎞")

        stage_start = time.time()

        if not self.sharp_selector_script.exists():
            raise FileNotFoundError(f"Sharp frame selector not found: {self.sharp_selector_script}")

        cmd = [
            sys.executable,
            str(self.sharp_selector_script),
        ]
        # One or more media inputs (videos and/or image folders)
        cmd.extend(str(p) for p in self.media_inputs)

        # Output into data_dir/images
        cmd.extend(["--output", str(self.images_dir)])

        # Per-input options
        cmd.extend(["--mode", "auto"])
        cmd.extend(["--method", self.sharp_method])
        cmd.extend(["--frame-step", str(max(1, self.sharp_frame_step))])

        if self.sharp_top_k and self.sharp_top_k > 0:
            cmd.extend(["--top-k", str(self.sharp_top_k)])
        if self.sharp_min_score is not None:
            cmd.extend(["--min-score", str(self.sharp_min_score)])

        # New temporal-aware options
        if self.sharp_temporal_method and self.sharp_temporal_method != "none":
            cmd.extend(["--temporal-method", self.sharp_temporal_method])
        if self.sharp_save_metrics_json:
            cmd.append("--save-metrics-json")
        if self.sharp_save_temporal_plot:
            cmd.append("--save-temporal-plot")

        if self.sharp_verbose:
            cmd.append("-" + "v" * self.sharp_verbose)

        console.print(f"🔧 Sharp-frame command: {' '.join(cmd)}", style="dim")
        console.print()

        subprocess.run(cmd, check=True)

        self.timings["sharp_frames"] = time.time() - stage_start
        console.print(
            f"\n✅ Sharp frame selection completed in {self.timings['sharp_frames']:.1f}s",
            style="bold green",
        )
        console.print(f"📂 Sharp frames saved in: {self.images_dir}", style="green")

    # -------------------------------------------------------------------------

    def _ensure_downsampled_images(self, factor: int) -> int:
        """
        Ensure a downsampled images_<factor>/ folder exists for GSplat.

        - If factor <= 1: return 1 (use original images/).
        - If images_<factor>/ already exists and is non-empty: return factor.
        - Otherwise, create images_<factor>/ by downsampling images/ with Pillow.

        Returns the actually used factor (1 or the requested factor).
        """
        if factor is None or factor <= 1:
            return 1

        src_dir = self.images_dir
        dst_dir = self.data_dir / f"images_{factor}"

        # Reuse if folder already exists and has images
        if dst_dir.exists():
            existing = sorted(
                [p for p in dst_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            )
            if existing:
                console.print(f"📂 Using existing downsampled folder: {dst_dir}", style="cyan")
                return factor

        # Otherwise (re)create it
        dst_dir.mkdir(parents=True, exist_ok=True)

        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "Pillow is required for automatic image downsampling. "
                "Install it with: pip install pillow"
            ) from e

        img_paths = sorted(
            [p for p in src_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if not img_paths:
            console.print(
                f"[yellow]⚠️ No source images found in {src_dir} when trying to build images_{factor}/.[/yellow]",
            )
            return 1

        console.print(
            f"🔻 Creating downsampled images (factor={factor}) in {dst_dir}...",
            style="cyan",
        )

        for img_path in img_paths:
            out_path = dst_dir / img_path.name
            if out_path.exists():
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    new_size = (max(1, w // factor), max(1, h // factor))
                    im_resized = im.resize(new_size, Image.Resampling.LANCZOS)
                    im_resized.save(out_path)
            except Exception as exc:
                console.print(
                    f"[yellow]⚠️ Failed to downsample {img_path.name}: {exc}[/yellow]"
                )

        # Sanity check: if nothing got written, fall back to factor 1
        final_imgs = list(dst_dir.glob("*.jpg")) + list(dst_dir.glob("*.jpeg")) + list(dst_dir.glob("*.png"))
        if not final_imgs:
            console.print(
                f"[yellow]⚠️ Downsampling produced no images in {dst_dir}. "
                f"Falling back to original images/ with factor 1.[/yellow]"
            )
            return 1

        console.print(f"✅ Downsampled images saved in: {dst_dir}", style="green")
        return factor

    # -------------------------------------------------------------------------

    def run_sfm(self) -> None:
        """Stage: Structure from Motion."""
        self._stage_header("sfm", "Structure from Motion (SfM)", "📷")

        stage_start = time.time()

        sfm_cfg = dict(self.preset.get("sfm", {}))  # copy

        cmd = [
            sys.executable,
            "scripts/run_hloc.py",
            "--project_path", str(self.data_dir),
        ]

        # Do not send custom-only keys as flags (data_factor is used later for GSplat)
        sfm_cfg_for_flags = dict(sfm_cfg)
        sfm_cfg_for_flags.pop("data_factor", None)

        # Explicit mapping from config keys → HLOC CLI flags
        hloc_flag_map = {
            # pair generation
            "matching_method": "--matching-method",
            "seq_overlap": "--seq-overlap",
            "seq_quadratic_overlap": "--seq-quadratic-overlap",
            "seq_loop_closure": "--seq-loop-closure",
            "num_matched": "--num_matched",

            # features / matchers
            "global_extractor": "--global_extractor",
            "extractor": "--extractor",
            "matcher": "--matcher",
            "superglue_weights": "--superglue_weights",
            "sinkhorn_iterations": "--sinkhorn_iterations",

            # camera / backend
            "camera_model": "--camera_model",
            "sfm_backend": "--sfm-backend",
            "glomap_bin": "--glomap-bin",

            # misc
            "vis": "--vis",
            "keep_intermediates": "--keep_intermediates",
            # if you ever add:
            "image_dir": "--image-dir",
        }

        # Convert config keys to proper HLOC flags
        for key, val in sfm_cfg_for_flags.items():
            if val is None:
                continue
            flag = hloc_flag_map.get(key)
            if flag is None:
                # Unknown / unmapped key: ignore rather than crashing
                continue

            # booleans -> flag only when True
            if isinstance(val, bool):
                if val:
                    cmd.append(flag)
                continue

            # list / tuple -> repeat
            if isinstance(val, (list, tuple)):
                for item in val:
                    cmd.extend([flag, str(item)])
                continue

            # scalar -> flag + value
            cmd.extend([flag, str(val)])

        # Optional extra raw flags (escape hatch)
        if self.hloc_flags:
            cmd.extend(self.hloc_flags)

        console.print(f"🔧 HLOC command: {' '.join(cmd)}", style="dim")
        console.print()

        subprocess.run(cmd, check=True)

        self.timings["sfm"] = time.time() - stage_start
        console.print(f"\n✅ SfM completed in {self.timings['sfm']:.1f}s", style="bold green")

    # -------------------------------------------------------------------------

    def validate_reconstruction(self) -> None:
        """
        Validate reconstruction as a separate step.

        - To avoid duplicate COLMAP reports, we only call parse_colmap.py here
          when SfM was skipped (i.e. reconstruction already existed).
        """
        console.print("\n" + "─" * 80, style="cyan")
        console.print("🔍 VALIDATION: Checking COLMAP reconstruction", style="bold cyan")
        console.print("─" * 80 + "\n", style="cyan")

        stage_start = time.time()

        if not self.sparse_dir.exists():
            raise FileNotFoundError(f"Reconstruction not found: {self.sparse_dir}")

        # If SfM just ran via run_hloc.py, that script already called parse_colmap.py
        # in your setup, so we skip re-running it to avoid duplicate reports.
        if self.args.skip_sfm:
            try:
                cmd = [sys.executable, "./scripts/parse_colmap.py", str(self.sparse_dir)]
                subprocess.run(cmd, check=True)
            except FileNotFoundError:
                console.print("⚠️  parse_colmap.py not found, skipping", style="yellow")
        else:
            console.print(
                "ℹ️  Reconstruction was already analyzed in the SfM stage; "
                "skipping duplicate COLMAP report.",
                style="dim",
            )

        self.timings["validation"] = time.time() - stage_start
        console.print(f"\n✅ Validation completed in {self.timings['validation']:.1f}s", style="bold green")

    # -------------------------------------------------------------------------

    def run_depth(self) -> None:
        """Stage: Generate COLMAP-aligned depth maps."""
        self._stage_header("depth", "Generating Depth Maps", "🌊")

        stage_start = time.time()

        if not self.sparse_dir.exists():
            raise FileNotFoundError(f"Sparse reconstruction not found: {self.sparse_dir}")

        depth_script = Path("scripts") / "gen_aligned_depths.py"
        if not depth_script.exists():
            raise FileNotFoundError(f"Depth script not found: {depth_script}")

        cmd = [
            sys.executable,
            str(depth_script),
            "--colmap_sparse_path", str(self.sparse_dir),
            "--image_dir", str(self.images_dir),
            "--output_dir", str(self.depth_dir),
            "--model_type", self.args.depth_model,
            "--device", self.args.depth_device,
        ]

        console.print(f"🔧 Depth command: {' '.join(cmd)}", style="dim")
        console.print()

        subprocess.run(cmd, check=True)

        self.timings["depth"] = time.time() - stage_start
        console.print(f"\n✅ Depth generation completed in {self.timings['depth']:.1f}s", style="bold green")
        console.print(f"📂 Depth maps saved in: {self.depth_dir}", style="green")

    # -------------------------------------------------------------------------

    def run_gsplat(self) -> None:
        """Stage: Train 3D Gaussian Splatting."""
        self._stage_header("training", "Training 3D Gaussian Splatting", "🎨")

        stage_start = time.time()

        # GSplat is hardcoded: <repo_root>/src/gsplat
        gsplat_dir = (Path(__file__).parent.parent / "src" / "gsplat").resolve()
        trainer_script = gsplat_dir / "examples" / "simple_trainer.py"
        if not trainer_script.exists():
            raise FileNotFoundError(f"GSplat trainer not found at: {trainer_script}")

        gs_cfg = dict(self.preset.get("gsplat", {}))  # copy
        strategy_cfg = dict(gs_cfg.get("strategy", {})) if isinstance(gs_cfg.get("strategy"), dict) else {}
        # pull out depth_lambda (used only when with_depth is enabled)
        depth_lambda = gs_cfg.pop("depth_lambda", 0.1)
        # we will not pass these as flags directly:
        for k in ("strategy",):
            gs_cfg.pop(k, None)

        # Determine subcommand: mcmc or default
        subcommand = gs_cfg.pop("subcommand", None)
        if subcommand is None:
            # backward-compatibility: if config had 'mode' or something similar
            subcommand = "mcmc"

        cmd = [
            sys.executable,
            str(trainer_script),
            str(subcommand),
        ]

        # Required paths (never overridden by config)
        cmd.extend(["--data-dir", str(self.data_dir)])
        cmd.extend(["--result-dir", str(self.result_dir)])

        # Data factor: prefer gsplat.data_factor, else sfm.data_factor, else 4.
        # Also ensure the corresponding images_<factor>/ folder exists for GSplat.
        sfm_cfg = self.preset.get("sfm", {})
        data_factor = gs_cfg.pop("data_factor", sfm_cfg.get("data_factor", 4))
        data_factor = self._ensure_downsampled_images(data_factor)
        cmd.extend(["--data-factor", str(data_factor)])

        # Map remaining gs_cfg keys to flags
        extend_cmd_from_config(cmd, "", gs_cfg)
        # Map strategy.* keys
        if strategy_cfg:
            extend_cmd_from_config(cmd, "strategy.", strategy_cfg)

        # Depth-guided training flags (only if depth stage is enabled)
        if self.args.with_depth:
            cmd.append("--depth-loss")
            cmd.extend(["--depth-lambda", str(depth_lambda)])

        # Save ply can also be forced by CLI default
        if getattr(self.args, "save_ply", True) and "--save-ply" not in cmd and "--save_ply" not in cmd:
            cmd.append("--save-ply")

        # Raw pass-through flags
        if self.gs_flags:
            cmd.extend(self.gs_flags)

        env = os.environ.copy()
        if self.args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

        console.print(f"🔧 GSplat command: {' '.join(cmd)}", style="dim")
        console.print(f"🎮 GPU: {env.get('CUDA_VISIBLE_DEVICES', 'default')}", style="dim")
        console.print()

        try:
            subprocess.run(cmd, env=env, cwd=str(gsplat_dir), check=True)
        except subprocess.CalledProcessError as e:
            console.print(f"\n❌ Training failed with exit code {e.returncode}", style="bold red")
            console.print(f"\nTry running manually to see full error:", style="yellow")
            console.print(f"  cd {gsplat_dir}", style="dim")
            console.print(f"  {' '.join(cmd)}", style="dim")
            raise

        self.timings["training"] = time.time() - stage_start
        console.print(f"\n✅ Training completed in {self.timings['training']:.1f}s", style="bold green")

        # Optionally launch GSplat simple_viewer on the latest checkpoint
        if self.with_gsplat_viewer:
            ckpt_path = self._find_latest_ckpt()
            if ckpt_path is None:
                console.print(
                    f"[yellow]⚠️ --with-gsplat-viewer was requested, but no checkpoint was found in {self.result_dir}.[/yellow]"
                )
            else:
                self.start_gsplat_viewer(ckpt_path)

    # -------------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print final summary."""
        total_time = time.time() - (self.start_time or time.time())

        console.print("\n" + "=" * 80, style="bold green")
        console.print("🎉 PIPELINE COMPLETED!", style="bold green")
        console.print("=" * 80 + "\n", style="bold green")

        table = Table(title="⏱️  Timing Summary")
        table.add_column("Stage", style="cyan")
        table.add_column("Time", justify="right", style="green")

        if "sharp_frames" in self.timings:
            table.add_row("Sharp frames", f"{self.timings['sharp_frames']:.1f}s")
        if "sfm" in self.timings:
            table.add_row("SfM", f"{self.timings['sfm']:.1f}s")
        if "validation" in self.timings:
            table.add_row("Validation", f"{self.timings['validation']:.1f}s")
        if "depth" in self.timings:
            table.add_row("Depth", f"{self.timings['depth']:.1f}s")
        if "training" in self.timings:
            table.add_row("Training", f"{self.timings['training']:.1f}s")
        table.add_row("Total", f"{total_time:.1f}s ({total_time/60:.1f} min)", style="bold")

        console.print(table)
        console.print(f"\n📂 Results: {self.result_dir}", style="green")
        console.print()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified 3DGS Pipeline: Images → SfM → Depth → Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto mode (recommended for beginners)
  python run_full_pipeline.py --data_dir ./data/garden

  # Manual preset
  python run_full_pipeline.py --data_dir ./data/garden --preset balanced

  # External config
  python run_full_pipeline.py --data_dir ./data/garden --config configs/balanced.yaml

  # Symbolic overrides (no quotes)
  python run_full_pipeline.py --data_dir ./data/garden --config balanced \\
      --hloc-override matching_method=sequential \\
      --gs-override batch_size=2
        """,
    )

    parser.add_argument("--data_dir", type=str, required=True, help="Data directory with images/")

    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="balanced",
        help="Quality preset (ignored if --config is used)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (or name in configs/, e.g. 'balanced')",
    )
    parser.add_argument(
        "--auto-preset",
        action="store_true",
        help="Auto-select preset based on dataset (ignored if --config is used)",
    )
    parser.add_argument("--result_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--skip-sfm", action="store_true", help="Skip SfM stage")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    parser.add_argument(
        "--save-ply",
        action="store_true",
        default=True,
        help="Save .ply (also honored from gsplat.save_ply in config)",
    )

    # Depth-related flags
    parser.add_argument(
        "--with-depth",
        action="store_true",
        help="Generate COLMAP-aligned depth maps and enable depth-guided GSplat (--depth-loss/--depth-lambda).",
    )
    parser.add_argument(
        "--depth-model",
        default="depth_anything_v2",
        choices=["depth_anything_v2", "zoedepth", "midas"],
        help="Depth model to use for depth map generation.",
    )
    parser.add_argument(
        "--depth-device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for depth inference.",
    )
    parser.add_argument(
        "--depth-dir",
        type=str,
        default=None,
        help="Optional custom directory for depth maps (default: <data_dir>/depths).",
    )

    # Sharp-frame preprocessing from raw media
    parser.add_argument(
        "--media",
        nargs="+",
        default=None,
        help=(
            "Optional raw media inputs (video files and/or image folders). "
            "If provided and images/ does not exist, sharp frames will be "
            "extracted into <data_dir>/images before SfM."
        ),
    )
    parser.add_argument(
        "--sharp-method",
        choices=["laplacian", "sobel", "canny", "combined"],
        default="combined",
        help="Sharpness metric for pre-SfM frame selection when using --media.",
    )
    parser.add_argument(
        "--sharp-top-k",
        type=int,
        default=0,
        help=(
            "Max frames/images to keep per media input when using --media. "
            "0 or negative = keep all above min-score."
        ),
    )
    parser.add_argument(
        "--sharp-min-score",
        type=float,
        default=None,
        help="Minimum sharpness score to keep when using --media.",
    )
    parser.add_argument(
        "--sharp-frame-step",
        type=int,
        default=1,
        help="Analyze every N-th frame in video inputs when using --media.",
    )
    parser.add_argument(
        "--sharp-selector-script",
        type=str,
        default="scripts/sharp_frame_selector.py",
        help="Path to sharp_frame_selector.py utility script.",
    )
    parser.add_argument(
        "--sharp-verbose",
        action="count",
        default=0,
        help="Verbosity for sharp frame selector (-v, -vv).",
    )
    parser.add_argument(
        "--sharp-temporal-method",
        type=str,
        default="none",
        choices=["none", "uniform", "adaptive", "hybrid"],
        help=(
            "Temporal selection strategy for videos in sharp_frame_selector "
            "('none' = pure sharpness ranking)."
        ),
    )
    parser.add_argument(
        "--sharp-save-metrics-json",
        action="store_true",
        help="If set, pass --save-metrics-json to sharp_frame_selector.py for per-video JSON metrics.",
    )
    parser.add_argument(
        "--sharp-save-temporal-plot",
        action="store_true",
        help="If set, pass --save-temporal-plot to sharp_frame_selector.py for per-video temporal plots.",
    )

    # Viewer / visualization flags
    parser.add_argument(
        "--with-viewer",
        action="store_true",
        help="Launch the Viser SfM+GSplat viewer in parallel.",
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=8080,
        help="Port for the Viser viewer (default: 8080).",
    )
    parser.add_argument(
        "--viewer-poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval (seconds) for the viewer to reload SfM/PLY updates.",
    )

    parser.add_argument(
        "--with-gsplat-viewer",
        action="store_true",
        help=(
            "After training (or when --skip-training), launch GSplat simple_viewer "
            "on the latest checkpoint."
        ),
    )
    parser.add_argument(
        "--gs-viewer-port",
        type=int,
        default=8081,
        help="Port for GSplat simple_viewer (default: 8081).",
    )

    # Symbolic overrides: key=value (no quotes required)
    parser.add_argument(
        "--hloc-override",
        action="append",
        default=[],
        help="Override HLOC/SfM config entries (e.g. matching_method=sequential). "
             "Can be used multiple times.",
    )
    parser.add_argument(
        "--gs-override",
        action="append",
        default=[],
        help="Override GSplat config entries (e.g. batch_size=2, strategy.cap_max=2000000, depth_lambda=0.05). "
             "Can be used multiple times.",
    )

    # Raw pass-through flags (still allowed)
    parser.add_argument(
        "--hloc-flags",
        type=str,
        default="",
        help="Extra raw flags passed to scripts/run_hloc.py (single quoted string).",
    )
    parser.add_argument(
        "--gs-flags",
        type=str,
        default="",
        help="Extra raw flags passed to GSplat simple_trainer (single quoted string).",
    )

    args = parser.parse_args()
    pipeline = Pipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()