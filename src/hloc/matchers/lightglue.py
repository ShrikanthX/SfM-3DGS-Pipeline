# hloc/matchers/lightglue.py

import sys
from pathlib import Path

from .. import logger
from ..utils.base_model import BaseModel

# this file is at: <root>/hloc/matchers/lightglue.py
this_file = Path(__file__).resolve()
project_root = this_file.parents[2]          # <root> (AutoHLOC_working)
lightglue_path = project_root / "third_party" / "LightGlue"
sys.path.append(str(lightglue_path))

from lightglue import LightGlue as LG  # noqa: E402


class LightGlue(BaseModel):
    default_conf = {
        "match_threshold": 0.2,
        "filter_threshold": 0.2,
        "width_confidence": 0.99,
        "depth_confidence": 0.95,
        "features": "superpoint",
        # we’ll try to load this from third_party/LightGlue/weights
        "model_name": "superpoint_lightglue.pth",
        "flash": True,
        "mp": False,
        "add_scale_ori": False,
    }

    required_inputs = [
        "image0",
        "keypoints0",
        "scores0",
        "descriptors0",
        "image1",
        "keypoints1",
        "scores1",
        "descriptors1",
    ]

    def _init(self, conf):
        model_name = conf.get("model_name", self.default_conf["model_name"])

        weights_dir = lightglue_path / "weights"
        local_weights = weights_dir / model_name

        logger.info(f"Loading LightGlue matcher (requested weights: {model_name})")

        # keep match_threshold → filter_threshold
        if "match_threshold" in conf and "filter_threshold" not in conf:
            conf["filter_threshold"] = conf["match_threshold"]

        if local_weights.exists():
            conf["weights"] = str(local_weights)
            logger.info(f"Found local LightGlue weights at: {local_weights}")
        else:
            logger.warning(
                "LightGlue weights file not found locally "
                f"({local_weights}). Continuing without explicit weights – "
                "will rely on LightGlue's own defaults."
            )

        self.net = LG(**conf)
        logger.info("LightGlue model ready.")

    def _forward(self, data):
        input0 = {
            "image": data["image0"],
            "keypoints": data["keypoints0"],
            "descriptors": data["descriptors0"].permute(0, 2, 1),
        }
        if "scales0" in data:
            input0["scales"] = data["scales0"]
        if "oris0" in data:
            input0["oris"] = data["oris0"]

        input1 = {
            "image": data["image1"],
            "keypoints": data["keypoints1"],
            "descriptors": data["descriptors1"].permute(0, 2, 1),
        }
        if "scales1" in data:
            input1["scales"] = data["scales1"]
        if "oris1" in data:
            input1["oris"] = data["oris1"]

        return self.net({"image0": input0, "image1": input1})
