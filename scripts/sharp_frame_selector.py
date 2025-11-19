#!/usr/bin/env python3
"""
sharp_frame_selector.py

Select sharp frames from one or more videos and/or image folders using classical
and composite CV sharpness metrics (Laplacian variance, Sobel/gradient magnitude,
Canny edge density, and a simple high-frequency FFT energy term).

Designed for 3D Gaussian Splatting / NeRF dataset preparation.

Key features
------------
- Supports videos and image folders (or a mix of both).
- Multiple sharpness methods: laplacian, sobel, canny, combined (composite).
- Optional temporal-aware selection for videos: uniform / adaptive / hybrid.
- Optional JSON metrics + temporal plots per video for analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Set, Any

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("sharp_frame_selector")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def setup_logging(verbosity: int = 0) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )


def resize_max(image: np.ndarray, max_size: int | None) -> np.ndarray:
    if max_size is None:
        return image
    h, w = image.shape[:2]
    scale = max(h, w) / float(max_size)
    if scale <= 1.0:
        return image
    new_w = int(round(w / scale))
    new_h = int(round(h / scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def center_crop(image: np.ndarray, crop_factor: float) -> np.ndarray:
    """Crop to center region; crop_factor in (0, 1] is fraction of width/height."""
    if crop_factor >= 1.0:
        return image
    if crop_factor <= 0.0:
        raise ValueError("crop_factor must be > 0 if used")
    h, w = image.shape[:2]
    ch = int(round(h * crop_factor))
    cw = int(round(w * crop_factor))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return image[y0: y0 + ch, x0: x0 + cw]


# -----------------------------------------------------------------------------
# Sharpness metrics
# -----------------------------------------------------------------------------

def compute_sharpness(
    gray: np.ndarray,
    method: str = "laplacian",
    crop_factor: float = 0.7,
) -> Dict[str, float]:
    """Compute sharpness metrics on a grayscale image.

    Returns a dict with individual metric values and a "score" for selection.

    Metrics:
        - laplacian_var: variance of Laplacian.
        - sobel_std: std-dev of gradient magnitude (Sobel).
        - grad_mean: mean of gradient magnitude.
        - canny_density: fraction of edge pixels.
        - fft_high_freq_energy: log-sum of high-frequency FFT magnitudes.
        - score: the scalar used for ranking / selection.
    """
    roi = center_crop(gray, crop_factor) if crop_factor < 1.0 else gray

    metrics: Dict[str, float] = {}

    # Laplacian variance
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    metrics["laplacian_var"] = float(lap.var())

    # Sobel / gradient magnitude
    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    metrics["sobel_std"] = float(np.std(grad_mag))
    metrics["grad_mean"] = float(np.mean(grad_mag))

    # Canny edge density
    edges = cv2.Canny(roi, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)
    metrics["canny_density"] = edge_density

    # High-frequency FFT energy
    # (simple "band-stop" of low frequencies around DC)
    try:
        rows, cols = roi.shape[:2]
        fft = np.fft.fft2(roi)
        fft_shift = np.fft.fftshift(fft)
        crow, ccol = rows // 2, cols // 2
        cutoff_ratio = 0.1
        r = int(min(rows, cols) * cutoff_ratio)
        mask = np.ones((rows, cols), np.uint8)
        mask[crow - r:crow + r, ccol - r:ccol + r] = 0
        fft_shift_filtered = fft_shift * mask
        magnitude_spectrum = np.log(np.abs(fft_shift_filtered) + 1.0)
        metrics["fft_high_freq_energy"] = float(np.sum(magnitude_spectrum))
    except Exception as e:  # FFT is best-effort; don't fail if something odd happens
        logger.debug("FFT high-frequency energy failed: %s", e)
        metrics["fft_high_freq_energy"] = 0.0

    # ------------------------------------------------------------------
    # Score selection
    # ------------------------------------------------------------------
    if method == "laplacian":
        metrics["score"] = metrics["laplacian_var"]
    elif method == "sobel":
        metrics["score"] = metrics["sobel_std"]
    elif method == "canny":
        metrics["score"] = metrics["canny_density"]
    elif method == "combined":
        # Composite score inspired by TemporalAwareBlurDetector.calculate_composite_score
        # Heuristic normalisations into ~[0, 1]
        lap_n = min(metrics["laplacian_var"] / 1000.0, 1.0)
        sob_n = min(metrics["sobel_std"] / 2000.0, 1.0)
        grad_n = min(metrics["grad_mean"] / 20.0, 1.0)
        can_n = min(metrics["canny_density"] / 0.1, 1.0)
        fft_n = min(metrics["fft_high_freq_energy"] / 10000.0, 1.0)

        metrics["score"] = (
            0.3 * lap_n
            + 0.2 * sob_n
            + 0.2 * grad_n
            + 0.2 * can_n
            + 0.1 * fft_n
        )
    else:
        raise ValueError(f"Unknown sharpness method: {method}")

    return metrics


def iter_image_files(root: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


# -----------------------------------------------------------------------------
# Image & video analysis
# -----------------------------------------------------------------------------

def analyze_images(
    image_root: Path,
    method: str,
    crop_factor: float,
    resize_to: int | None,
) -> List[Tuple[float, Path, Dict[str, float]]]:
    scores: List[Tuple[float, Path, Dict[str, float]]] = []
    for img_path in tqdm(list(iter_image_files(image_root)), desc=f"Images {image_root}"):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Failed to read image %s", img_path)
            continue
        img = resize_max(img, resize_to)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        metrics = compute_sharpness(gray, method=method, crop_factor=crop_factor)
        scores.append((metrics["score"], img_path, metrics))
    return scores


def analyze_video(
    video_path: Path,
    method: str,
    crop_factor: float,
    resize_to: int | None,
    frame_step: int,
) -> Tuple[List[Tuple[float, int, float, Dict[str, float]]], float, int]:
    """Return per-frame metrics list, FPS and total frame count.

    scores list entries:
        (score, frame_index, timestamp_seconds, metrics_dict)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    scores: List[Tuple[float, int, float, Dict[str, float]]] = []

    frame_idx = 0
    step = max(1, frame_step)

    with tqdm(total=total_frames // step + 1, desc=f"Video {video_path.name}") as tq:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            t_sec = frame_idx / fps
            frame_small = resize_max(frame, resize_to)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            metrics = compute_sharpness(gray, method=method, crop_factor=crop_factor)
            scores.append((metrics["score"], frame_idx, t_sec, metrics))

            frame_idx += 1
            tq.update(1)

    cap.release()
    return scores, fps, total_frames


# -----------------------------------------------------------------------------
# Selection helpers
# -----------------------------------------------------------------------------

def select_top_k(
    scored_items: List[Tuple[float, Any, Dict[str, float]]],
    min_score: float | None,
    top_k: int | None,
) -> List[Any]:
    """Generic top-k selection from (score, id, metrics) tuples."""
    if not scored_items:
        return []

    # Filter by min_score if requested
    if min_score is not None:
        scored_items = [x for x in scored_items if x[0] >= min_score]

    scored_items.sort(key=lambda x: x[0], reverse=True)

    if top_k is not None and top_k > 0:
        scored_items = scored_items[:top_k]

    keep_ids = [id_ for (_, id_, _) in scored_items]
    return keep_ids


class TemporalFrameSelector:
    """Temporal-aware selection strategies for video frames."""

    def select_frames(
        self,
        frames_data: List[Dict[str, Any]],
        target_count: int,
        method: str = "uniform",
        min_sharpness: float | None = None,
    ) -> List[int]:
        """Select frame indices considering both sharpness and temporal distribution.

        frames_data elements:
            {
                "frame_index": int,
                "timestamp": float,
                "score": float,
                "metrics": Dict[str, float],
            }
        """
        if target_count is None or target_count <= 0:
            # Keep all frames (respecting min_sharpness if given)
            if min_sharpness is None:
                return [f["frame_index"] for f in frames_data]
            return [f["frame_index"] for f in frames_data if f["score"] >= min_sharpness]

        if not frames_data:
            return []

        # Filter by sharpness first
        if min_sharpness is not None:
            sharp_frames = [f for f in frames_data if f["score"] >= min_sharpness]
        else:
            sharp_frames = list(frames_data)

        if len(sharp_frames) <= target_count:
            return sorted(f["frame_index"] for f in sharp_frames)

        timestamps = np.array([f["timestamp"] for f in sharp_frames], dtype=float)
        scores = np.array([f["score"] for f in sharp_frames], dtype=float)

        method = (method or "uniform").lower()
        if method == "uniform":
            selected = self._uniform_selection(sharp_frames, timestamps, target_count)
        elif method == "adaptive":
            selected = self._adaptive_selection(sharp_frames, timestamps, scores, target_count)
        elif method == "hybrid":
            selected = self._hybrid_selection(sharp_frames, timestamps, scores, target_count)
        else:
            # Fallback: just do uniform
            selected = self._uniform_selection(sharp_frames, timestamps, target_count)

        return sorted(f["frame_index"] for f in selected)

    @staticmethod
    def _uniform_selection(
        frames: List[Dict[str, Any]],
        timestamps: np.ndarray,
        target_count: int,
    ) -> List[Dict[str, Any]]:
        """Select frames uniformly distributed in time."""
        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        frames_sorted = [frames[i] for i in order]

        total_duration = timestamps[-1] - timestamps[0]
        if target_count <= 1 or total_duration <= 0:
            # Degenerate case: just take the best-scoring frames in order
            return frames_sorted[:target_count]

        target_interval = total_duration / float(target_count - 1)

        selected_indices: List[int] = []
        current_time = timestamps[0]

        for _ in range(target_count):
            time_diffs = np.abs(timestamps - current_time)
            best_idx = int(np.argmin(time_diffs))
            if best_idx not in selected_indices:
                selected_indices.append(best_idx)
            current_time += target_interval
            if current_time > timestamps[-1]:
                break

        return [frames_sorted[i] for i in selected_indices]

    @staticmethod
    def _adaptive_selection(
        frames: List[Dict[str, Any]],
        timestamps: np.ndarray,
        scores: np.ndarray,
        target_count: int,
    ) -> List[Dict[str, Any]]:
        """Adaptive selection based on both time distribution and sharpness."""
        try:
            from sklearn.cluster import KMeans  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            logger.warning("sklearn not available for adaptive temporal selection (%s), "
                           "falling back to uniform.", e)
            return TemporalFrameSelector._uniform_selection(frames, timestamps, target_count)

        # Normalize timestamps and scores for clustering
        t0, t1 = float(timestamps.min()), float(timestamps.max())
        if t1 <= t0:
            return TemporalFrameSelector._uniform_selection(frames, timestamps, target_count)

        norm_timestamps = (timestamps - t0) / (t1 - t0 + 1e-8)
        s0, s1 = float(scores.min()), float(scores.max())
        if s1 <= s0:
            norm_scores = np.zeros_like(scores)
        else:
            norm_scores = (scores - s0) / (s1 - s0 + 1e-8)

        features = np.column_stack([norm_timestamps, norm_scores])

        k = min(target_count, len(frames))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(features)

        selected: List[Dict[str, Any]] = []
        for cluster_id in range(k):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            best_idx = cluster_indices[np.argmax(scores[cluster_indices])]
            selected.append(frames[best_idx])

        return selected

    @staticmethod
    def _hybrid_selection(
        frames: List[Dict[str, Any]],
        timestamps: np.ndarray,
        scores: np.ndarray,
        target_count: int,
    ) -> List[Dict[str, Any]]:
        """Hybrid approach: ensure minimum time gap while maximizing sharpness."""
        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        frames_sorted = [frames[i] for i in order]
        scores_sorted = scores[order]

        if len(frames_sorted) <= target_count:
            return frames_sorted

        total_duration = timestamps[-1] - timestamps[0]
        min_time_gap = total_duration / float(target_count * 2) if total_duration > 0 else 0.0

        selected: List[Dict[str, Any]] = []

        # Always include first and last for coverage if possible
        if frames_sorted:
            selected.append(frames_sorted[0])
        if len(frames_sorted) > 1:
            selected.append(frames_sorted[-1])

        remaining = [f for f in frames_sorted if f not in selected]

        def valid_gap(candidate_ts: float, selected_frames: List[Dict[str, Any]]) -> bool:
            for sf in selected_frames:
                if abs(candidate_ts - sf["timestamp"]) < min_time_gap:
                    return False
            return True

        while len(selected) < target_count and remaining:
            best_frame = None
            best_score = -1.0
            for f in remaining:
                ts = f["timestamp"]
                sc = f["score"]
                if valid_gap(ts, selected) and sc > best_score:
                    best_frame = f
                    best_score = sc

            if best_frame is None:
                # If no frame maintains the gap, just take the sharpest remaining
                best_frame = max(remaining, key=lambda x: x["score"])
            selected.append(best_frame)
            remaining.remove(best_frame)

        return selected


# -----------------------------------------------------------------------------
# Saving & metrics
# -----------------------------------------------------------------------------

def save_image_selections(
    scores: List[Tuple[float, Path, Dict[str, float]]],
    keep_paths: List[Path],
    out_dir: Path,
    csv_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["score", "source_path"] + sorted(scores[0][2].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for score, img_path, metrics in scores:
            if img_path in keep_paths:
                writer.writerow(
                    {
                        "score": score,
                        "source_path": str(img_path),
                        **metrics,
                    }
                )


def save_video_selections(
    video_path: Path,
    scores: List[Tuple[float, int, float, Dict[str, float]]],
    keep_indices: List[int],
    out_dir: Path,
    csv_path: Path,
    image_format: str = "jpg",
) -> None:
    """Re-open the video and save selected frames to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Video basename for naming
    stem = video_path.stem

    # Write CSV with metrics
    if scores:
        fieldnames = ["score", "frame_index", "timestamp", "filename"] + sorted(scores[0][3].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for score, frame_idx, t_sec, metrics in scores:
                if frame_idx in keep_indices:
                    filename = f"{stem}_{frame_idx:06d}.{image_format}"
                    writer.writerow(
                        {
                            "score": score,
                            "frame_index": frame_idx,
                            "timestamp": t_sec,
                            "filename": filename,
                            **metrics,
                        }
                    )

    # Now actually extract and save the frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video for saving frames: {video_path}")

    keep_set: Set[int] = set(keep_indices)
    frame_idx = 0
    with tqdm(total=max(keep_indices) + 1 if keep_indices else 0, desc=f"Save {stem}") as tq:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in keep_set:
                filename = f"{stem}_{frame_idx:06d}.{image_format}"
                out_path = out_dir / filename
                cv2.imwrite(str(out_path), frame)
                tq.update(1)
                keep_set.remove(frame_idx)
                if not keep_set:
                    break
            frame_idx += 1

    cap.release()


def log_quality_metrics_video(
    scores: List[Tuple[float, int, float, Dict[str, float]]],
    keep_indices: List[int],
    label: str,
) -> None:
    if not scores:
        return

    sel_set = set(keep_indices)
    metric_names = set(scores[0][3].keys())
    metric_names.add("score")

    logger.info("==== Video frame quality metrics (%s) ====", label)
    for name in sorted(metric_names):
        all_vals: List[float] = []
        sel_vals: List[float] = []
        for score, _, _, metrics in scores:
            val = metrics.get(name, score if name == "score" else 0.0)
            all_vals.append(val)
        for score, frame_idx, _, metrics in scores:
            if frame_idx in sel_set:
                val = metrics.get(name, score if name == "score" else 0.0)
                sel_vals.append(val)
        if not all_vals or not sel_vals:
            continue
        logger.info(
            "  %-24s all: mean=%.4f, std=%.4f | selected: mean=%.4f, std=%.4f",
            name,
            float(np.mean(all_vals)),
            float(np.std(all_vals)),
            float(np.mean(sel_vals)),
            float(np.std(sel_vals)),
        )


def log_quality_metrics_images(
    scores: List[Tuple[float, Path, Dict[str, float]]],
    keep_paths: List[Path],
    label: str,
) -> None:
    if not scores:
        return

    sel_set = set(keep_paths)
    metric_names = set(scores[0][2].keys())
    metric_names.add("score")

    logger.info("==== Image quality metrics (%s) ====", label)
    for name in sorted(metric_names):
        all_vals: List[float] = []
        sel_vals: List[float] = []
        for score, img_path, metrics in scores:
            val = metrics.get(name, score if name == "score" else 0.0)
            all_vals.append(val)
        for score, img_path, metrics in scores:
            if img_path in sel_set:
                val = metrics.get(name, score if name == "score" else 0.0)
                sel_vals.append(val)
        if not all_vals or not sel_vals:
            continue
        logger.info(
            "  %-24s all: mean=%.4f, std=%.4f | selected: mean=%.4f, std=%.4f",
            name,
            float(np.mean(all_vals)),
            float(np.std(all_vals)),
            float(np.mean(sel_vals)),
            float(np.std(sel_vals)),
        )


def generate_temporal_analysis_plot(
    all_frames: List[Dict[str, Any]],
    selected_frames: List[Dict[str, Any]],
    output_dir: Path,
    label: str,
) -> None:
    """Generate temporal distribution plot for a single video."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning("matplotlib not available for temporal plot (%s); skipping.", e)
        return

    if not all_frames:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    all_timestamps = [f["timestamp"] for f in all_frames]
    all_scores = [f["score"] for f in all_frames]

    sel_timestamps = [f["timestamp"] for f in selected_frames]
    sel_scores = [f["score"] for f in selected_frames]

    plt.figure(figsize=(12, 6))

    # Scatter of all vs selected
    plt.subplot(1, 2, 1)
    plt.scatter(all_timestamps, all_scores, alpha=0.4, s=15, label="All sampled")
    if sel_timestamps:
        plt.scatter(sel_timestamps, sel_scores, alpha=0.9, s=30, label="Selected", color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Sharpness score")
    plt.title(f"Temporal distribution: {label}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Intervals between selected frames
    plt.subplot(1, 2, 2)
    if len(sel_timestamps) > 1:
        sorted_ts = sorted(sel_timestamps)
        intervals = np.diff(sorted_ts)
        plt.bar(range(len(intervals)), intervals)
        plt.xlabel("Interval index")
        plt.ylabel("Time interval (s)")
        plt.title("Intervals between selected frames")
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Not enough frames for interval stats", ha="center", va="center")

    plt.tight_layout()
    plot_path = output_dir / "temporal_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Temporal analysis plot saved: %s", plot_path)


def calculate_temporal_stats(selected_frames: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute simple temporal statistics over selected frames."""
    if len(selected_frames) < 2:
        return {}
    timestamps = sorted(f["timestamp"] for f in selected_frames)
    intervals = np.diff(timestamps)
    if len(intervals) == 0:
        return {}
    return {
        "total_frames": len(selected_frames),
        "time_span": float(timestamps[-1] - timestamps[0]),
        "mean_interval": float(np.mean(intervals)),
        "std_interval": float(np.std(intervals)),
        "min_interval": float(np.min(intervals)),
        "max_interval": float(np.max(intervals)),
        "coverage_efficiency": float((timestamps[-1] - timestamps[0]) / max(len(selected_frames) - 1, 1)),
    }


def save_json_metrics_for_video(
    args: argparse.Namespace,
    video_path: Path,
    all_frames: List[Dict[str, Any]],
    selected_frames: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "sharp_frame_metrics.json"

    payload: Dict[str, Any] = {
        "video": str(video_path),
        "params": {
            "method": args.method,
            "crop_factor": args.crop_factor,
            "resize_to": args.resize,
            "frame_step": args.frame_step,
            "top_k": args.top_k,
            "min_score": args.min_score,
            "temporal_method": args.temporal_method,
        },
        "selected_frames": selected_frames,
        "all_frames": [
            {
                "frame_index": f["frame_index"],
                "timestamp": f["timestamp"],
                "score": f["score"],
            }
            for f in all_frames
        ],
        "temporal_stats": calculate_temporal_stats(selected_frames),
    }

    with metrics_path.open("w") as f:
        json.dump(payload, f, indent=2)

    logger.info("JSON metrics saved: %s", metrics_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract sharp frames from one or more videos and/or image folders for 3DGS/NeRF training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=str,
        help="One or more input paths (video files and/or image folders).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="sharp_frames",
        help=(
            "Output root directory to store selected frames. "
            "If multiple inputs are given, frames go to <output>/<stem>/."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["auto", "video", "images"],
        default="auto",
        help="Force processing mode for each input, or auto-detect per input.",
    )
    p.add_argument(
        "--method",
        choices=["laplacian", "sobel", "canny", "combined"],
        default="combined",
        help="Sharpness metric to use.",
    )
    p.add_argument(
        "--resize",
        type=int,
        default=640,
        help="Resize longest side to this size for analysis (0 = no resize).",
    )
    p.add_argument(
        "--crop-factor",
        type=float,
        default=0.7,
        help="Fraction of central crop used for sharpness evaluation (1.0 = no crop).",
    )
    p.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Analyze every N-th frame for videos.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help=(
            "Max frames/images to keep per input. "
            "0 or negative = keep all above min-score."
        ),
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum sharpness score to keep (applied before top-k).",
    )
    p.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Image format for saved frames.",
    )
    p.add_argument(
        "--temporal-method",
        type=str,
        default="none",
        choices=["none", "uniform", "adaptive", "hybrid"],
        help=(
            "Temporal selection strategy for videos. "
            "'none' = pure sharpness ranking (previous behaviour)."
        ),
    )
    p.add_argument(
        "--save-metrics-json",
        action="store_true",
        help="If set, save JSON metrics per video in the corresponding output folder.",
    )
    p.add_argument(
        "--save-temporal-plot",
        action="store_true",
        help="If set, save a temporal analysis PNG per video in the corresponding output folder.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v, -vv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    out_root = Path(args.output).expanduser().resolve()
    resize_to = None if args.resize <= 0 else args.resize

    for inp in args.inputs:
        input_path = Path(inp).expanduser().resolve()
        if not input_path.exists():
            raise SystemExit(f"Input does not exist: {input_path}")

        # Decide mode
        mode = args.mode
        if mode == "auto":
            if input_path.is_dir():
                mode = "images"
            else:
                mode = "video"

        # Output layout:
        # - If there is only a single input, write frames directly into the
        #   output root (flat layout). This is convenient for pipelines that
        #   expect images directly under e.g. data/<scene>/images/.
        # - If there are multiple inputs, keep the old behaviour and create
        #   one subfolder per input: <output>/<stem>/.
        if len(args.inputs) == 1:
            out_dir = out_root
        else:
            stem = input_path.stem if input_path.is_file() else input_path.name
            out_dir = out_root / stem
        csv_path = out_dir / "sharp_frames.csv"

        if mode == "images":
            scores = analyze_images(
                input_path,
                method=args.method,
                crop_factor=args.crop_factor,
                resize_to=resize_to,
            )
            # Map to (score, id, metrics)
            generic_scores = [(s, p, m) for (s, p, m) in scores]

            keep_paths_ids = select_top_k(
                scored_items=generic_scores,
                min_score=args.min_score,
                top_k=args.top_k if args.top_k and args.top_k > 0 else None,
            )
            keep_paths = [p for p in keep_paths_ids]  # already Paths

            logger.info(
                "Selected %d / %d images for %s",
                len(keep_paths),
                len(scores),
                input_path,
            )
            log_quality_metrics_images(scores, keep_paths, label=str(input_path))
            save_image_selections(scores, keep_paths, out_dir, csv_path)

        elif mode == "video":
            if not input_path.is_file():
                raise SystemExit(
                    f"Expected a video file for mode=video, got directory: {input_path}"
                )

            scores, fps, total_frames = analyze_video(
                input_path,
                method=args.method,
                crop_factor=args.crop_factor,
                resize_to=resize_to,
                frame_step=args.frame_step,
            )

            if not scores:
                logger.warning("No frames processed for video %s", input_path)
                continue

            # Build frames_data for metrics / temporal selection
            frames_data: List[Dict[str, Any]] = []
            for score, frame_idx, t_sec, metrics in scores:
                frames_data.append(
                    {
                        "frame_index": frame_idx,
                        "timestamp": t_sec,
                        "score": score,
                        "metrics": metrics,
                    }
                )

            # Decide selection strategy
            if args.temporal_method != "none" and args.top_k and args.top_k > 0:
                selector = TemporalFrameSelector()
                keep_indices = selector.select_frames(
                    frames_data,
                    target_count=args.top_k,
                    method=args.temporal_method,
                    min_sharpness=args.min_score,
                )
            else:
                generic_scores = [(s, idx, m) for (s, idx, _, m) in scores]
                keep_indices = select_top_k(
                    scored_items=generic_scores,
                    min_score=args.min_score,
                    top_k=args.top_k if args.top_k and args.top_k > 0 else None,
                )

            keep_indices = sorted(set(keep_indices))

            logger.info(
                "Selected %d / %d analysed frames (~%d / %d total) for %s",
                len(keep_indices),
                len(scores),
                len(scores) * max(1, args.frame_step),
                total_frames,
                input_path,
            )

            log_quality_metrics_video(scores, keep_indices, label=str(input_path))
            save_video_selections(
                input_path,
                scores,
                keep_indices,
                out_dir,
                csv_path,
                image_format=args.image_format,
            )

            # Optional JSON / plot
            if args.save_metrics_json or args.save_temporal_plot:
                selected_frames = [
                    f for f in frames_data if f["frame_index"] in set(keep_indices)
                ]
                if args.save_metrics_json:
                    save_json_metrics_for_video(
                        args,
                        input_path,
                        frames_data,
                        selected_frames,
                        out_dir,
                    )
                if args.save_temporal_plot:
                    generate_temporal_analysis_plot(
                        frames_data,
                        selected_frames,
                        out_dir,
                        label=input_path.name,
                    )

        else:
            raise SystemExit(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
