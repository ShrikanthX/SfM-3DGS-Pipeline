# glomap_integration.py
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from hloc import logger


def run_glomap(
    database_path: Path,
    output_dir: Path,
    colorize: bool = False,
) -> None:
    """
    Call the system-wide `glomap` you said you have:
        glomap mapper --database_path DATABASE --output_path MODEL
    We also normalize colors so it doesn’t look different from AutoHLOC logs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "glomap",
        "mapper",
        "--database_path",
        str(database_path),
        "--output_path",
        str(output_dir),
    ]

    logger.info("Running GLOMAP with command:\n" + " ".join(cmd))


    # run and fail loud if glomap errors
    subprocess.run(cmd, check=True, env=env)

    logger.info("✅ GLOMAP reconstruction finished.")
