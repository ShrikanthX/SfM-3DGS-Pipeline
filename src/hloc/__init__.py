import logging
from pathlib import Path

from packaging import version

__version__ = "1.5"

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("hloc")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO_ID = "local"

try:
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    DEVICE = "cpu"

try:
    import pycolmap
except ImportError:
    logger.warning("pycolmap is not installed, some features may not work.")
else:
    min_version = version.parse("0.6.0")
    found_version = pycolmap.__version__
    if found_version != "dev":
        found_v = version.parse(found_version)
        if found_v < min_version:
            s = f"pycolmap>={min_version}"
            logger.warning(
                "hloc requires %s but found pycolmap==%s, "
                'please upgrade with `pip install --upgrade "%s"`',
                s,
                found_version,
                s,
            )
