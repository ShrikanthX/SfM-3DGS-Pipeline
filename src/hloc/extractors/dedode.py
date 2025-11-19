# hloc/extractors/dedode.py

import sys
import tempfile
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image

from .. import logger
from ..utils.base_model import BaseModel

repo_root = Path(__file__).resolve().parents[2]
dedode_root = repo_root / "third_party" / "DeDoDe"
if str(dedode_root) not in sys.path:
    sys.path.insert(0, str(dedode_root))

from DeDoDe import dedode_detector_L, dedode_descriptor_B  # type: ignore


class DeDoDeExtractor(BaseModel):
    default_conf = {
        "name": "dedode",
        "max_keypoints": 5000,
        "descriptor": "B",
        "resize_max": 1600,
    }

    required_inputs = ["image", "name"]

    def _init(self, conf):
        self.max_keypoints = int(conf.get("max_keypoints", 5000))
        self.detector = dedode_detector_L(weights=None)
        desc_type = conf.get("descriptor", "B")
        if desc_type == "B":
            self.descriptor = dedode_descriptor_B(weights=None)
        else:
            from DeDoDe import dedode_descriptor_G  # type: ignore

            self.descriptor = dedode_descriptor_G(weights=None, dinov2_weights=None)

    @torch.no_grad()
    def _forward(self, data):
        image = data["image"]  # (1, C, H, W)
        image = image[0].cpu()
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        pil_img = to_pil_image(image)

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            pil_img.save(tmp.name)
            det = self.detector.detect_from_path(
                tmp.name, num_keypoints=self.max_keypoints
            )
            kpts = det["keypoints"]  # (N, 2)
            scores = det["confidence"]  # (N,)

            desc = self.descriptor.describe_keypoints_from_path(
                tmp.name, kpts
            )["descriptions"]  # (N, D)

        keypoints = torch.as_tensor(kpts, dtype=torch.float32).unsqueeze(0)
        scores = torch.as_tensor(scores, dtype=torch.float32).unsqueeze(0)
        descriptors = torch.as_tensor(desc, dtype=torch.float32).unsqueeze(0).permute(
            0, 2, 1
        )

        return {
            "keypoints": keypoints,
            "scores": scores,
            "descriptors": descriptors,
        }


extractor = DeDoDeExtractor
