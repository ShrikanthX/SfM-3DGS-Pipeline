#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import time
import logging
import sys
from pathlib import Path
from rich.console import Console
import argparse
import shutil
import subprocess

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    logger,
)


try:
    from hloc.pairs_from_retrieval import main as pairs_from_retrieval_main
except ImportError:  
    from hloc import pairs_from_retrieval as pairs_from_retrieval_main

try:
    from hloc.pairs_from_sequential import main as pairs_from_sequential_main
except ImportError:
    pairs_from_sequential_main = None  

try:
    from hloc.pairs_from_exhaustive import main as pairs_from_exhaustive_main
except ImportError:
    pairs_from_exhaustive_main = None


# set defaults
DEFAULT_GLOBAL = "netvlad"
DEFAULT_EXTRACTOR = "superpoint_aachen"
DEFAULT_MATCHER = "superglue"
DEFAULT_NUM_MATCHED = 10
DEFAULT_CAMERA = "PINHOLE"
DEFAULT_SUPERGLUE_WEIGHTS = "outdoor"
DEFAULT_SINKHORN = 50
DEFAULT_MATCHING_METHOD = "retrieval"

console = Console()
logger = logging.getLogger("hloc")


def run_parser_on_model(sfm_dir: Path):
    sparse_dir = sfm_dir  
    parser_script = Path(__file__).parent / "parse_colmap.py"
    if parser_script.exists():
        subprocess.run(
            ["python", str(parser_script), str(sparse_dir)],
            check=False,  
        )


def _discover_hloc_things():
    """Ask hloc what is registered right now."""
    all_extractors = sorted(extract_features.confs.keys())
    all_matchers = sorted(match_features.confs.keys())
    all_global = [
        name
        for name, conf in extract_features.confs.items()
        if conf.get("output", "").startswith("global-feats-")
    ]
    return all_extractors, all_matchers, sorted(all_global)


def build_parser() -> argparse.ArgumentParser:
    try:
        from rich_argparse import RichHelpFormatter as Formatter
    except Exception:
        Formatter = argparse.ArgumentDefaultsHelpFormatter

    all_extractors, all_matchers, all_global = _discover_hloc_things()

    ge_default = DEFAULT_GLOBAL if DEFAULT_GLOBAL in all_global else (all_global[0] if all_global else None)
    ex_default = DEFAULT_EXTRACTOR if DEFAULT_EXTRACTOR in all_extractors else all_extractors[0]
    m_default = DEFAULT_MATCHER if DEFAULT_MATCHER in all_matchers else all_matchers[0]

    parser = argparse.ArgumentParser(
        usage="run_hloc.py [OPTIONS]",
        formatter_class=Formatter,
        epilog=(
            "examples:\n"
            "  python run_hloc.py --project_path ./data/garden "
            "--extractor superpoint_aachen --matcher superglue --global_extractor netvlad\n"
            "  python run_hloc.py --project_path ./data/garden "
            "--extractor disk --matcher disk-lightglue --num_matched 15 --sfm-backend fastmap\n"
        ),
    )

    # required
    parser.add_argument(
        "--project_path",
        type=Path,
        required=True,
        help="project root (should contain images/)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="override images dir (default: PROJECT_PATH/images)",
    )

    # matching method (retrieval / sequential / exhaustive)
    parser.add_argument(
        "--matching-method",
        choices=["retrieval", "sequential", "exhaustive"],
        default=DEFAULT_MATCHING_METHOD,
        help="how to build image pairs (default: retrieval)",
    )

    # sequential-specific knobs (only used if matching-method=sequential)
    parser.add_argument(
        "--seq-overlap",
        type=int,
        default=10,
        help="number of neighbor images to pair in sequential mode (default: 10)",
    )
    parser.add_argument(
        "--seq-quadratic-overlap",
        action="store_true",
        help="use quadratic overlap for sequential mode",
    )
    parser.add_argument(
        "--seq-loop-closure",
        action="store_true",
        help="in sequential mode, also add retrieval-based loop-closure pairs (needs global feats)",
    )

    # retrieval / global model
    parser.add_argument(
        "--global_extractor",
        choices=all_global,
        default=ge_default,
        help=f"Image-retrieval model for pair generation (default: {ge_default})",
    )

    # local features
    parser.add_argument(
        "--extractor",
        choices=all_extractors,
        default=ex_default,
        help=f"Local feature extractor to use (default: {ex_default})",
    )

    # matcher
    parser.add_argument(
        "--matcher",
        choices=all_matchers,
        default=m_default,
        help=f"Feature matcher to use (default: {m_default})",
    )

    # superglue extras
    parser.add_argument(
        "--superglue_weights",
        choices=["outdoor", "indoor"],
        default=DEFAULT_SUPERGLUE_WEIGHTS,
        help=f"Weights for SuperGlue (only used if --matcher=superglue) (default: {DEFAULT_SUPERGLUE_WEIGHTS})",
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        default=DEFAULT_SINKHORN,
        help=f"Number of Sinkhorn iterations for SuperGlue (default: {DEFAULT_SINKHORN})",
    )

    # generic SfM params
    parser.add_argument(
        "--num_matched",
        type=int,
        default=DEFAULT_NUM_MATCHED,
        help=f"number of retrieved/matched images per query (default: {DEFAULT_NUM_MATCHED})",
    )
    parser.add_argument(
        "--camera_model",
        choices=["OPENCV", "OPENCV_FISHEYE", "EQUIRECTANGULAR", "PINHOLE", "SIMPLE_PINHOLE"],
        default=DEFAULT_CAMERA,
        help=f"camera model to pass to COLMAP / FastMap / GLOMAP (default: {DEFAULT_CAMERA})",
    )

    # SfM selector
    parser.add_argument(
        "--sfm-backend",
        choices=["colmap", "fastmap", "glomap"],
        default="fastmap",
        help="which SfM backend to run after HLOC (default: fastmap)",
    )
    parser.add_argument(
        "--glomap-bin",
        default="glomap",
        help="path to the glomap executable (default: glomap)",
    )

    # visualization (FastMap viewer)
    parser.add_argument(
        "--vis",
        action="store_true",
        help="after reconstruction, launch FastMap viewer on the final sparse model",
    )

    parser.add_argument(
        "--keep_intermediates",
        action="store_true",
        help="do not delete intermediate sfm outputs",
    )

    return parser


def _list_image_names(images_dir: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    names = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            names.append(p.name)
    return names


def fastmap_viewer(model_dir: Path):
    """
    Launch `python -m fastmap.vis <model_dir>` from the local ./fastmap folder,
    but do it fully detached so:
      - it doesn't block run_hloc.py
      - its logs don't clutter our terminal
      - closing the window does not affect the shell
    """
    repo_root = Path(__file__).parent
    fastmap_root = repo_root / "fastmap"
    if not fastmap_root.exists():
        logger.warning("FastMap folder not found at %s — cannot launch viewer.", fastmap_root)
        return

    cmd = [sys.executable, "-m", "fastmap.vis", str(model_dir)]

    try:
        # On POSIX: start_new_session=True makes it its own process group.
        subprocess.Popen(
            cmd,
            cwd=str(fastmap_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,   # Python ≥3.2
        )
        logger.info("Launched FastMap viewer for %s (detached)", model_dir)
    except Exception as e:
        logger.error("Failed to launch FastMap viewer: %s", e)


def run_sfm_pipeline(
    *,
    project_path: Path,
    images_dir: Path,
    matching_method: str,
    seq_overlap: int,
    seq_quadratic: bool,
    seq_loop_closure: bool,
    extractor_name: str,
    matcher_name: str,
    global_extractor: str,
    superglue_weights: str,
    sinkhorn_iterations: int,
    num_matched: int,
    camera_model: str,
    keep_intermediates: bool,
    sfm_backend: str,
    glomap_bin: str,
    vis: bool,
):
    logger.info("🚀 Starting AutoHLOC SfM pipeline...")
    out_dir = project_path / "hloc_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 1
    retrieval_path = None
    step_times = {}
    t0 = time.time()

    # 1) only make global feats if we actually need them
    t = time.time()
    need_global = (matching_method == "retrieval") or (matching_method == "sequential" and seq_loop_closure)
    if need_global:
        logger.info(f"Step {step}/5: extracting global features using {global_extractor}...")
        retrieval_conf = extract_features.confs[global_extractor]
        retrieval_path = extract_features.main(retrieval_conf, images_dir, out_dir)
        step += 1
        step_times["global_features"] = time.time() - t

    # 2) local features (always)
    t = time.time()
    logger.info(f"Step {step}/5: extracting local features using {extractor_name}...")
    feature_conf = extract_features.confs[extractor_name]
    features_path = extract_features.main(feature_conf, images_dir, out_dir)
    step += 1
    step_times["local_features"] = time.time() - t

    # 3) pairs
    t = time.time()
    pairs_path = out_dir / "pairs.txt"
    if matching_method == "retrieval":
        logger.info(f"Step {step}/5: generating image pairs with retrieval (top-{num_matched})...")
        assert retrieval_path is not None, "retrieval_path should exist in retrieval mode"
        pairs_from_retrieval_main(retrieval_path, pairs_path, num_matched=num_matched)
    elif matching_method == "sequential":
        if pairs_from_sequential_main is None:
            raise RuntimeError("pairs_from_sequential.py not found, but --matching-method=sequential was used.")
        logger.info(
            f"Step {step}/5: generating sequential pairs (overlap={seq_overlap}, "
            f"quadratic={seq_quadratic}, loop_closure={seq_loop_closure})..."
        )
        img_names = _list_image_names(images_dir)
        kwargs = dict(
            output=pairs_path,
            image_list=img_names,
            overlap=seq_overlap,
            quadratic_overlap=seq_quadratic,
        )
        if seq_loop_closure:
            kwargs["features"] = retrieval_path
        pairs_from_sequential_main(**kwargs)
    elif matching_method == "exhaustive":
        if pairs_from_exhaustive_main is None:
            raise RuntimeError("pairs_from_exhaustive.py not found, but --matching-method=exhaustive was used.")
        logger.info(f"Step {step}/5: generating exhaustive pairs...")
        img_names = _list_image_names(images_dir)
        pairs_from_exhaustive_main(
            output=pairs_path,
            image_list=img_names,
        )
    else:
        raise ValueError(f"Unknown matching method: {matching_method}")
    step += 1
    step_times["pair_generation"] = time.time() - t

    # 4) matching
    t = time.time()
    logger.info(f"Step {step}/5: matching features using {matcher_name}...")
    matcher_conf = match_features.confs[matcher_name]
    if matcher_name == "superglue":
        matcher_conf = {
            "output": "matches-superglue",
            "model": {
                "name": "superglue",
                "weights": superglue_weights,
                "sinkhorn_iterations": sinkhorn_iterations,
            },
        }
    matches_path = match_features.main(
        matcher_conf,
        pairs_path,
        feature_conf["output"],
        out_dir,
    )
    step += 1
    step_times["feature_matching"] = time.time() - t

    # 5) reconstruction
    t = time.time()
    logger.info(f"Step {step}/5: running 3D reconstruction...")
    sfm_dir = out_dir / "sfm"
    sfm_dir.mkdir(parents=True, exist_ok=True)

    # decide backend 
    use_fastmap = sfm_backend == "fastmap"
    use_glomap = sfm_backend == "glomap"

    reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=images_dir,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        camera_model=camera_model,
        verbose=True,
        use_fastmap=use_fastmap,
        use_glomap=use_glomap,
        glomap_bin=glomap_bin,
    )

    # move final sparse to project/sparse/0
    final_sparse = project_path / "sparse" / "0"
    final_sparse.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        src = sfm_dir / name
        if src.exists():
            shutil.move(str(src), final_sparse / name)

    if not keep_intermediates:
        for item in sfm_dir.iterdir():
            if item.is_file():
                item.unlink()

    logger.info(f"✅ Reconstruction saved to {final_sparse}")

    # analyze it
    run_parser_on_model(final_sparse)

    #launch FastMap viewer
    if vis:
        fastmap_viewer(final_sparse)

    step_times["reconstruction"] = time.time() - t
    total = time.time() - t0
    step_times["total"] = total
    print_timer_summary(step_times, project_path)


def print_timer_summary(step_times: dict, project_path: str | Path):
    console.print("\n[bold cyan]⏱ Pipeline timing summary[/bold cyan]")
    console.print(f"[dim]project: {project_path}[/dim]")
    for k, v in step_times.items():
        console.print(f"  • {k:20s}: {v:7.2f} s")
    console.print(f"[bold]Total[/bold]: {step_times['total']:.2f} s\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_path = args.project_path.resolve()
    images_dir = args.image_dir or (project_path / "images")

    run_sfm_pipeline(
        project_path=project_path,
        images_dir=images_dir,
        matching_method=args.matching_method,
        seq_overlap=args.seq_overlap,
        seq_quadratic=args.seq_quadratic_overlap,
        seq_loop_closure=args.seq_loop_closure,
        extractor_name=args.extractor,
        matcher_name=args.matcher,
        global_extractor=args.global_extractor,
        superglue_weights=args.superglue_weights,
        sinkhorn_iterations=args.sinkhorn_iterations,
        num_matched=args.num_matched,
        camera_model=args.camera_model,
        keep_intermediates=args.keep_intermediates,
        sfm_backend=args.sfm_backend,
        glomap_bin=args.glomap_bin,
        vis=args.vis,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
