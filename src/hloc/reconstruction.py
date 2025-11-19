import logging
import shutil
import multiprocessing
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

import pycolmap

from . import logger
from .utils.database import COLMAPDatabase
from .triangulation import (
    import_features,
    import_matches,
    estimation_and_geometric_verification,
)


def create_empty_db(database_path: Path):
    """Create an empty COLMAP database."""
    if database_path.exists():
        logger.warning(f'The database {database_path} already exists, deleting it.')
        database_path.unlink()
    logger.info('Creating an empty database...')
    db = COLMAPDatabase.connect(str(database_path))
    db.create_tables()
    db.commit()
    db.close()


def import_images(
    image_dir: Path,
    database_path: Path,
    camera_mode: pycolmap.CameraMode,
    camera_model: str,
    image_list: Optional[List[str]] = None,
):
    """Import images into the database without nesting or duplicating folders."""
    logger.info('Importing images into the database...')
    options = pycolmap.ImageReaderOptions(camera_model=camera_model)

    valid_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_dir = Path(image_dir)
    image_paths = sorted([p for p in image_dir.glob("*") if p.suffix in valid_exts])
    image_names = [p.name for p in image_paths]

    logger.debug(f"Found {len(image_names)} images for import.")

    # this is the C++/pycolmap call that actually writes images into the DB
    with pycolmap.ostream():
        pycolmap.import_images(
            str(database_path),
            str(image_dir.resolve()),
            camera_mode,
            image_names=image_list or image_names,
            options=options,
        )


def get_image_ids(database_path: Path) -> Dict[str, int]:
    db = COLMAPDatabase.connect(str(database_path))
    images = {name: image_id for name, image_id in db.execute("SELECT name, image_id FROM images;")}
    db.close()
    return images


def run_mapper(
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    mapper_options: Optional[Dict[str, Any]] = None,
) -> Optional[pycolmap.Reconstruction]:
    models_path = sfm_dir / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    logger.info("Running 3D reconstruction (COLMAP incremental mapper)...")
    if mapper_options is None:
        mapper_options = {}

    options = pycolmap.IncrementalPipelineOptions({
        "num_threads": min(multiprocessing.cpu_count(), 16),
        **mapper_options,
    })

    with pycolmap.ostream():
        reconstructions = pycolmap.incremental_mapping(
            str(database_path), str(image_dir), str(models_path), options
        )

    if not reconstructions:
        logger.error("Could not reconstruct any model!")
        return None

    largest_index = max(reconstructions, key=lambda i: reconstructions[i].num_reg_images())
    logger.info(
        f"Largest model is #{largest_index} with {reconstructions[largest_index].num_reg_images()} images."
    )

    for filename in ["images.bin", "cameras.bin", "points3D.bin"]:
        if (sfm_dir / filename).exists():
            (sfm_dir / filename).unlink()
        shutil.move(str(models_path / str(largest_index) / filename), str(sfm_dir))

    return reconstructions[largest_index]


def _hoist_glomap_outputs(sfm_dir: Path) -> bool:
    """Find cameras.bin/images.bin/points3D.bin in subfolders and move to sfm_dir."""
    wanted = {"cameras.bin", "images.bin", "points3D.bin"}

    # already at root?
    root_have = {p.name for p in sfm_dir.glob("*.bin")}
    if wanted.issubset(root_have):
        return True

    # otherwise search one level deep
    for sub in sfm_dir.iterdir():
        if not sub.is_dir():
            continue
        sub_bins = {p.name for p in sub.glob("*.bin")}
        if wanted.issubset(sub_bins):
            for name in wanted:
                src = sub / name
                dst = sfm_dir / name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
            return True

    return False


def run_glomap(
    *,
    sfm_dir: Path,
    database_path: Path,
    image_dir: Path,
    glomap_bin: str = "glomap",
):
    sfm_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        glomap_bin,
        "mapper",
        "--database_path",
        str(database_path),
        "--output_path",
        str(sfm_dir),
        "--image_path",    str(image_dir),
    ]
    logger.info("Running GLOMAP mapper...")
    logger.info(" ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("✅ GLOMAP reconstruction succeeded.")

    ok = _hoist_glomap_outputs(sfm_dir)
    if not ok:
        logger.error("GLOMAP finished but did not produce cameras.bin/images.bin/points3D.bin at the top level.")


def main(
    sfm_dir: Path,
    image_dir: Path,
    pairs: Path,
    features: Path,
    matches: Path,
    camera_model: str = 'SIMPLE_RADIAL',
    verbose: bool = False,
    mapper_options: Optional[Dict[str, Any]] = None,
    use_fastmap: bool = False,   # kept only so existing callsites don't break
    use_glomap: bool = False,
    glomap_bin: str = "glomap",
) -> Optional[pycolmap.Reconstruction]:
    from . import triangulation  # avoid circular import

    camera_mode = pycolmap.CameraMode.SINGLE
    skip_geometric_verification = use_fastmap or use_glomap

    sfm_dir.mkdir(parents=True, exist_ok=True)

    # 👇 key change: for GLOMAP we create a *different* DB file
    if use_glomap:
        database = sfm_dir / "glomap_database.db"
    else:
        database = sfm_dir / "database.db"

    # 1) create DB and import images
    create_empty_db(database)
    try:
        import_images(image_dir, database, camera_mode, camera_model)
    except RuntimeError as e:
        # this is the error you got: SQLite error: disk I/O error
        logger.error(f"pycolmap.import_images failed: {e}")
        raise

    image_ids = get_image_ids(database)

    # 2) import feats & matches
    triangulation.import_features(image_ids, database, features)
    triangulation.import_matches(
        image_ids,
        database,
        pairs,
        matches,
        skip_geometric_verification=skip_geometric_verification,
    )

    # 3) geo verification (colmap-only)
    if not skip_geometric_verification:
        triangulation.estimation_and_geometric_verification(database, pairs, verbose)

    # 4) backend
    if use_fastmap:
        logger.info("Running triangulation with FastMap instead of COLMAP...")
        reconstruction = triangulation.run_triangulation(
            model_path=sfm_dir,
            database_path=database,
            image_dir=image_dir,
            reference_model=None,
            verbose=verbose,
            sfm_backend="fastmap",
            camera_model=camera_model,
        )
        if reconstruction is not None:
            logger.info(f"Reconstruction statistics:\n{reconstruction.summary()}")
        else:
            logger.info("FastMap reconstruction completed successfully (no pycolmap summary available).")
        return reconstruction

    if use_glomap:
        logger.info("Running GLOMAP instead of COLMAP...")
        run_glomap(
            sfm_dir=sfm_dir,
            database_path=database,
            image_dir=image_dir,
            glomap_bin=glomap_bin,
        )
        return None

    # default: COLMAP
    reconstruction = run_mapper(sfm_dir, database, image_dir, mapper_options)
    if reconstruction is not None:
        logger.info(f"Reconstruction statistics:\n{reconstruction.summary()}")
    else:
        logger.info("Reconstruction finished but no model was created.")
    return reconstruction
