from pathlib import Path
import torch
from .. import logger
from ..utils.base_model import BaseModel


def _to_tensor(x, device, dtype=torch.float32):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def _nearest_index(src_kpts, query_kpts):
    # src_kpts: (N,2), query_kpts: (M,2)
    diff = query_kpts[:, None, :] - src_kpts[None, :, :]
    dist2 = (diff ** 2).sum(-1)
    return dist2.argmin(dim=1)


class XFeatLightGlue(BaseModel):
    default_conf = {
        "max_keypoints": 8000,
    }
    required_inputs = ["image0", "image1"]

    

    def _init(self, conf):
                # load xfeat from local accelerated_features clone
        root = Path(__file__).resolve().parents[3]
        local_repo = root / "third_party" / "accelerated_features"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xfeat = torch.hub.load(
            str(local_repo),
            "XFeat",
            source="local",
            pretrained=True,
            top_k=self.conf["max_keypoints"],
            trust_repo=True,
        )
        logger.info("Load XFeat(dense) model done.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xfeat = self.xfeat.to(self.device).eval()

    def _forward(self, data):
        device = self.device

        img0 = data["image0"].to(device)
        img1 = data["image1"].to(device)

        # get H, W for image_size that match_lighterglue needs
        if img0.ndim == 4:
            _, _, H0, W0 = img0.shape
        else:
            _, H0, W0 = img0.shape
        if img1.ndim == 4:
            _, _, H1, W1 = img1.shape
        else:
            _, H1, W1 = img1.shape

        # detect+describe
        out0 = self.xfeat.detectAndCompute(img0, top_k=self.conf["max_keypoints"])[0]
        out1 = self.xfeat.detectAndCompute(img1, top_k=self.conf["max_keypoints"])[0]
        out0["image_size"] = [H0, W0]
        out1["image_size"] = [H1, W1]

        # original keypoints (may have batch dim)
        kpts0_t = _to_tensor(out0["keypoints"], device)
        kpts1_t = _to_tensor(out1["keypoints"], device)
        if kpts0_t.ndim == 3 and kpts0_t.shape[0] == 1:
            kpts0_t = kpts0_t.squeeze(0)
        if kpts1_t.ndim == 3 and kpts1_t.shape[0] == 1:
            kpts1_t = kpts1_t.squeeze(0)

        # empty guard
        if kpts0_t.numel() == 0 or kpts1_t.numel() == 0:
            matches0 = torch.full((kpts0_t.shape[0],), -1, dtype=torch.int32, device=device)
            matches1 = torch.full((kpts1_t.shape[0],), -1, dtype=torch.int32, device=device)
            matching_scores0 = torch.zeros((kpts0_t.shape[0],), dtype=torch.float32, device=device)
            return {
                "keypoints0": kpts0_t.unsqueeze(0),
                "keypoints1": kpts1_t.unsqueeze(0),
                "matches0": matches0.unsqueeze(0),
                "matches1": matches1.unsqueeze(0),
                "matching_scores0": matching_scores0.unsqueeze(0),
            }

        # run lighterglue from verlab
        match_out = self.xfeat.match_lighterglue(out0, out1)
        if len(match_out) == 3:
            mkpts0, mkpts1, mconf = match_out
        else:
            mkpts0, mkpts1 = match_out
            mconf = None

        mkpts0_t = _to_tensor(mkpts0, device)
        mkpts1_t = _to_tensor(mkpts1, device)
        if mconf is None:
            mconf_t = torch.ones((mkpts0_t.shape[0],), device=device, dtype=torch.float32)
        else:
            mconf_t = _to_tensor(mconf, device).view(-1)

        # coord -> index
        if mkpts0_t.numel() == 0:
            matches0 = torch.full((kpts0_t.shape[0],), -1, dtype=torch.int32, device=device)
            matches1 = torch.full((kpts1_t.shape[0],), -1, dtype=torch.int32, device=device)
            matching_scores0 = torch.zeros((kpts0_t.shape[0],), dtype=torch.float32, device=device)
        else:
            idx0 = _nearest_index(kpts0_t, mkpts0_t)
            idx1 = _nearest_index(kpts1_t, mkpts1_t)

            N0 = kpts0_t.shape[0]
            N1 = kpts1_t.shape[0]
            matches0 = torch.full((N0,), -1, dtype=torch.int32, device=device)
            matches1 = torch.full((N1,), -1, dtype=torch.int32, device=device)
            matches0[idx0] = idx1
            matches1[idx1] = idx0

            # scores must be length N0, aligned with matches0
            matching_scores0 = torch.zeros((N0,), dtype=torch.float32, device=device)
            # fill scores only for matched indices
            matching_scores0[idx0] = mconf_t

        return {
            "keypoints0": kpts0_t.unsqueeze(0),
            "keypoints1": kpts1_t.unsqueeze(0),
            "matches0": matches0.unsqueeze(0),
            "matches1": matches1.unsqueeze(0),
            "matching_scores0": matching_scores0.unsqueeze(0),
        }
