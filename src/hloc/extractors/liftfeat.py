import sys
from pathlib import Path
from ..utils.base_model import BaseModel
from .. import logger

liftfeat_path = Path(__file__).parent / "../../third_party/LiftFeat"
sys.path.insert(0, str(liftfeat_path))

from models.liftfeat_wrapper import LiftFeat  # noqa: E402


class Liftfeat(BaseModel):
    """
    LiftFeat wrapper for HLoc-like interface.
    Expects weights to be present locally at:
        third_party/LiftFeat/weights/LiftFeat.pth
    Adjust `model_name` below if your file is named differently.
    """

    default_conf = {
        "keypoint_threshold": 0.05,
        "max_keypoints": 5000,
        # this is relative to third_party/LiftFeat/weights/
        "model_name": "LiftFeat.pth",
    }

    required_inputs = ["image"]

    def _init(self, conf):
        logger.info("Loading LiftFeat model...")

        # build local path to the weight file
        weights_dir = liftfeat_path / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_path = weights_dir / self.conf["model_name"]

        if not model_path.exists():
            raise FileNotFoundError(
                f"LiftFeat weight not found at {model_path}. "
                "Put your LiftFeat.pth there or change model_name in the config."
            )

        # LiftFeat wrapper wants numpy image, but we just init it here
        self.net = LiftFeat(
            weight=str(model_path),
            detect_threshold=self.conf["keypoint_threshold"],
            top_k=self.conf["max_keypoints"],
        )

        logger.info("Loading LiftFeat model done!")

    def _forward(self, data):
        # input comes as torch BCHW in [0,1]
        image = data["image"].cpu().numpy().squeeze() * 255.0
        # to HWC
        image = image.transpose(1, 2, 0)

        pred = self.net.extract(image)

        keypoints = pred["keypoints"]          # (N, 2)
        descriptors = pred["descriptors"]      # (N, D) torch
        scores = pred["scores"]                # (N,) torch

        # enforce max_keypoints if needed
        max_kp = self.conf["max_keypoints"]
        if max_kp > 0 and len(keypoints) > max_kp:
            # scores is torch, keypoints is numpy -> pick on torch side
            topk_scores, topk_idx = scores.topk(max_kp)
            keypoints = keypoints[topk_idx.cpu().numpy(), :2]
            descriptors = descriptors[topk_idx]
            scores = topk_scores

        # hloc expects batch dimension and descriptors as (B, D, N)
        return {
            "keypoints": keypoints[None],                     # (1, N, 2)
            "descriptors": descriptors[None].permute(0, 2, 1),  # (1, D, N)
            "scores": scores[None],                           # (1, N)
        }
