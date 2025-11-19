from pathlib import Path
import torch

from hloc import logger

from ..utils.base_model import BaseModel


class XFeat(BaseModel):
    default_conf = {
        "max_keypoints": 5000,
    }
    required_inputs = ["image"]

    def _init(self, conf):
        root = Path(__file__).resolve().parents[3]  
        local_repo = root / "third_party" / "accelerated_features"


        self.net = torch.hub.load(
            str(local_repo),
            "XFeat",
            source="local",
            pretrained=True,
            top_k=self.conf["max_keypoints"],
        )
        logger.info("Load XFeat(sparse) model from local accelerated_features done.")

    def _forward(self, data):
        pred = self.net.detectAndCompute(
            data["image"], top_k=self.conf["max_keypoints"]
        )[0]
        return {
            "keypoints": pred["keypoints"][None],
            "scores": pred["scores"][None],
            "descriptors": pred["descriptors"].T[None],
        }
