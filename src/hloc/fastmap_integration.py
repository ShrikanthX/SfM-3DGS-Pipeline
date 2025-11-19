# hloc/fastmap_integration.py
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def run_fastmap(
    database_path: Path,
    image_dir: Path,
    model_path: Path,
    *,
    headless: bool = True,
    calibrated: bool = True,
    device: str = "cuda:0",
    pinhole: bool = False,
):
    """
    Run FastMap engine on an existing COLMAP-style database (database_path).

    Args:
        database_path: Path to HLOC-generated database.db file.
        image_dir: Path to image directory.
        model_path: Path to the SfM working directory (contains database.db).
        headless: Run FastMap without viewer.
        calibrated: Use existing camera intrinsics from DB.
        device: Device string for FastMap (e.g., 'cuda:0').
        pinhole: If True, enforce pinhole (no radial distortion) in FastMap.

    Steps:
      1) create a clean 'fastmap_out' under model_path
      2) run FastMap
      3) move {cameras,images,points3D}.bin from fastmap_out/sparse/0 into model_path
    """
    database_path = Path(database_path)
    image_dir = Path(image_dir)
    model_path = Path(model_path)
    fastmap_out = model_path / "fastmap_out"

    model_path.mkdir(parents=True, exist_ok=True)
    if fastmap_out.exists():
        logger.info("Removing previous FastMap outputs at: %s", fastmap_out)
        shutil.rmtree(fastmap_out)

    # Path to FastMap runner inside the repo
    repo_root = Path(__file__).resolve().parent.parent
    fastmap_script = repo_root / "fastmap" / "run.py"
    if not fastmap_script.exists():
        raise FileNotFoundError(f"FastMap script not found at {fastmap_script}")

    # Build the FastMap CLI command
    cmd = [
        "python", str(fastmap_script),
        "--database", str(database_path),
        "--image_dir", str(image_dir),
        "--output_dir", str(fastmap_out),
        "--device", device,
    ]
    if headless:
        cmd.append("--headless")
    if calibrated:
        cmd.append("--calibrated")
    if pinhole:
        cmd.append("--pinhole")

    logger.info("Running FastMap with command:\n%s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("FastMap failed with return code %s", e.returncode)
        raise

    sparse_dir = fastmap_out / "sparse" / "0"
    if not sparse_dir.exists():
        raise RuntimeError(f"Expected FastMap output directory not found: {sparse_dir}")

    produced_files = []
    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        src = sparse_dir / fname
        dst = model_path / fname
        if src.exists():
            if dst.exists():
                logger.info("Overwriting existing %s", dst)
                dst.unlink()
            shutil.move(str(src), str(dst))
            produced_files.append(dst)
            logger.info("Moved %s → %s", src, dst)
        else:
            logger.warning("Expected FastMap output %s not found in %s", fname, sparse_dir)

    if not produced_files:
        raise RuntimeError(f"FastMap did not produce any expected outputs in {sparse_dir}")

    logger.info("✅ FastMap reconstruction succeeded. Outputs moved to %s", model_path)
    return produced_files
