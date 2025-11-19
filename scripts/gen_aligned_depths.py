from pathlib import Path
import sys

# -----------------------------------------------------------------------------
# Project roots & Python path setup
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

# Your existing src/ (for consistency with the rest of the project)
sys.path.insert(0, str(ROOT / "src"))

# Depth-Anything-V2 repo (so `import depth_anything_v2` works without pip install)
DA2_ROOT = ROOT / "third_party" / "Depth-Anything-V2"
if DA2_ROOT.exists():
    sys.path.insert(0, str(DA2_ROOT))

import numpy as np
import torch
import cv2
from typing import Dict, Tuple, Optional
import struct
from dataclasses import dataclass
import argparse
from tqdm import tqdm


# =============================================================================
# COLMAP STRUCTURES & READER
# =============================================================================


@dataclass
class CameraParams:
    """Camera parameters from COLMAP."""

    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    """Image data from COLMAP."""

    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass
class Point3D:
    """3D point from COLMAP sparse reconstruction."""

    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


class COLMAPReader:
    """Read COLMAP sparse reconstruction (binary format)."""

    @staticmethod
    def read_cameras_binary(path: Path) -> Dict[int, CameraParams]:
        """Read cameras.bin file."""
        cameras = {}
        with open(path, "rb") as f:
            num_cameras = struct.unpack("Q", f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack("I", f.read(4))[0]
                model_id = struct.unpack("I", f.read(4))[0]
                width = struct.unpack("Q", f.read(8))[0]
                height = struct.unpack("Q", f.read(8))[0]

                CAMERA_MODELS = {
                    0: "SIMPLE_PINHOLE",
                    1: "PINHOLE",
                    2: "SIMPLE_RADIAL",
                    3: "RADIAL",
                    4: "OPENCV",
                }
                model_name = CAMERA_MODELS.get(model_id, f"MODEL_{model_id}")

                num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}.get(model_id, 0)
                params = np.array(struct.unpack(f"{num_params}d", f.read(8 * num_params)))

                cameras[camera_id] = CameraParams(
                    camera_id, model_name, width, height, params
                )
        return cameras

    @staticmethod
    def read_images_binary(path: Path) -> Dict[int, Image]:
        """Read images.bin file."""
        images = {}
        with open(path, "rb") as f:
            num_images = struct.unpack("Q", f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack("I", f.read(4))[0]
                qvec = np.array(struct.unpack("4d", f.read(32)))
                tvec = np.array(struct.unpack("3d", f.read(24)))
                camera_id = struct.unpack("I", f.read(4))[0]

                name_bytes = b""
                while True:
                    char = f.read(1)
                    if char == b"\x00":
                        break
                    name_bytes += char
                name = name_bytes.decode("utf-8")

                num_points2D = struct.unpack("Q", f.read(8))[0]
                xys = np.zeros((num_points2D, 2))
                point3D_ids = np.zeros(num_points2D, dtype=np.int64)

                for i in range(num_points2D):
                    xy = struct.unpack("2d", f.read(16))
                    xys[i] = xy
                    point3D_ids[i] = struct.unpack("q", f.read(8))[0]

                images[image_id] = Image(
                    image_id, qvec, tvec, camera_id, name, xys, point3D_ids
                )
        return images

    @staticmethod
    def read_points3D_binary(path: Path) -> Dict[int, Point3D]:
        """Read points3D.bin file."""
        points3D = {}
        with open(path, "rb") as f:
            num_points = struct.unpack("Q", f.read(8))[0]
            for _ in range(num_points):
                point3D_id = struct.unpack("Q", f.read(8))[0]
                xyz = np.array(struct.unpack("3d", f.read(24)))
                rgb = np.array(struct.unpack("3B", f.read(3)))
                error = struct.unpack("d", f.read(8))[0]

                track_length = struct.unpack("Q", f.read(8))[0]
                track_data = struct.unpack(f"{2 * track_length}I", f.read(8 * track_length))
                image_ids = np.array(track_data[0::2])
                point2D_idxs = np.array(track_data[1::2])

                points3D[point3D_id] = Point3D(
                    point3D_id, xyz, rgb, error, image_ids, point2D_idxs
                )
        return points3D


# =============================================================================
# DEPTH ESTIMATOR
# =============================================================================


class DepthEstimator:
    """
    Unified interface for depth models.

    Supported model_type options:
      - "depth_anything_v2"      → official Depth-Anything-V2 repo
      - "depth_anything_v2_hf"   → HuggingFace Transformers backend
      - "zoedepth"               → ZoeDepth via torch.hub
      - "midas"                  → MiDaS via torch.hub
    """

    def __init__(
        self,
        model_type: str = "depth_anything_v2",
        device: str = "cuda",
        da2_encoder: str = "vits",
        da2_ckpt_path: Optional[str] = None,
    ):
        self.device = device
        self.model_type = model_type
        self.da2_encoder = da2_encoder
        self.da2_ckpt_path = Path(da2_ckpt_path) if da2_ckpt_path is not None else None
        self._uses_da2_infer = False  # flag for official DepthAnythingV2 backend

        self.model = self._load_model()

        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def _load_model(self):
        """Load the specified depth estimation model."""

        # ------------------------------------------------------------------
        # 1) Official Depth Anything V2 (local checkpoint, recommended)
        # ------------------------------------------------------------------
        if self.model_type == "depth_anything_v2":
            try:
                from depth_anything_v2.dpt import DepthAnythingV2
            except ImportError as exc:
                raise ImportError(
                    "Could not import DepthAnythingV2. Make sure the official "
                    "Depth-Anything-V2 repo is available, e.g. at:\n"
                    "  third_party/Depth-Anything-V2\n"
                    "and that it is on PYTHONPATH.\n\n"
                    f"Original error: {exc}"
                ) from exc

            model_configs = {
                "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
            }

            if self.da2_encoder not in model_configs:
                raise ValueError(
                    f"Unknown DepthAnythingV2 encoder '{self.da2_encoder}'. "
                    f"Supported: {list(model_configs.keys())}."
                )

            cfg = model_configs[self.da2_encoder]

            # Resolve checkpoint path
            if self.da2_ckpt_path is not None:
                ckpt = self.da2_ckpt_path
            else:
                repo_root = ROOT / "third_party" / "Depth-Anything-V2"
                ckpt = repo_root / "checkpoints" / f"depth_anything_v2_{self.da2_encoder}.pth"

            if not ckpt.exists():
                raise FileNotFoundError(
                    f"DepthAnythingV2 checkpoint not found at:\n  {ckpt}\n\n"
                    f"Download 'depth_anything_v2_{self.da2_encoder}.pth' from the official "
                    f"Depth-Anything-V2 repo and place it there, or pass --da2-ckpt-path."
                )

            model = DepthAnythingV2(**cfg)
            state = torch.load(str(ckpt), map_location="cpu")
            model.load_state_dict(state)

            # IMPORTANT: keep model in float32 to match DepthAnythingV2.infer_image expectations
            model = model.to(self.device).eval()

            self._uses_da2_infer = True
            return model

        # ------------------------------------------------------------------
        # 2) HuggingFace Transformers backend for Depth Anything V2
        # ------------------------------------------------------------------
        if self.model_type == "depth_anything_v2_hf":
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            if self.da2_ckpt_path is not None and self.da2_ckpt_path.exists():
                model_name = str(self.da2_ckpt_path)
            else:
                model_name = "depth-anything/Depth-Anything-V2-Small"

            try:
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModelForDepthEstimation.from_pretrained(model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Depth-Anything-V2 HF model from '{model_name}'.\n\n"
                    f"For offline usage, download the model with transformers and point "
                    f"--da2-ckpt-path to that folder (containing config.json, "
                    f"preprocessor_config.json, etc.).\n\n"
                    f"Original error: {exc}"
                ) from exc

            model = model.to(self.device)
            model.eval()
            if self.device == "cuda":
                model = model.half()
            return model

        # ------------------------------------------------------------------
        # 3) ZoeDepth
        # ------------------------------------------------------------------
        if self.model_type == "zoedepth":
            model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
            model = model.to(self.device)
            model.eval()

            if self.device == "cuda":
                model = model.half()

            return model

        # ------------------------------------------------------------------
        # 4) MiDaS
        # ------------------------------------------------------------------
        if self.model_type == "midas":
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
            model = model.to(self.device)
            model.eval()

            if self.device == "cuda":
                model = model.half()

            return model

        raise ValueError(f"Unknown model type: {self.model_type}")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth map from an RGB image."""

        # Official DepthAnythingV2 backend
        if self.model_type == "depth_anything_v2" and self._uses_da2_infer:
            # Depth-Anything-V2 examples use BGR (cv2.imread), so convert back
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            depth = self.model.infer_image(img_bgr)
            return depth.astype(np.float32)

        # HuggingFace Transformers backend
        if self.model_type == "depth_anything_v2_hf":
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.device == "cuda":
                inputs = {
                    k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()
                }

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            depth = prediction.squeeze().cpu().float().numpy()
            return depth

        # ZoeDepth
        if self.model_type == "zoedepth":
            from PIL import Image as PILImage

            pil_image = PILImage.fromarray(image)
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                depth = self.model.infer_pil(pil_image)
            return depth.astype(np.float32)

        # MiDaS
        if self.model_type == "midas":
            input_batch = self.transform(image).to(self.device)
            if self.device == "cuda":
                input_batch = input_batch.half()

            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            depth = prediction.squeeze().cpu().float().numpy()
            return depth

        raise ValueError(f"Unsupported model_type in predict(): {self.model_type}")


# =============================================================================
# DEPTH ALIGNMENT (COLMAP → METRIC)
# =============================================================================


class DepthAligner:
    """Align monocular depth maps to COLMAP metric scale."""

    @staticmethod
    def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        qvec = qvec / np.linalg.norm(qvec)
        w, x, y, z = qvec
        return np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
            ]
        )

    @staticmethod
    def get_intrinsic_matrix(camera: CameraParams) -> np.ndarray:
        """Extract intrinsic matrix K from camera parameters."""
        if camera.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            f, cx, cy = camera.params[:3]
            K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        elif camera.model in ["PINHOLE", "OPENCV", "RADIAL"]:
            fx, fy, cx, cy = camera.params[:4]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        else:
            raise ValueError(f"Unsupported camera model: {camera.model}")
        return K

    @staticmethod
    def create_sparse_depth_map(
        image: Image,
        camera: CameraParams,
        points3D: Dict[int, Point3D],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sparse depth map from COLMAP 3D points."""
        h, w = camera.height, camera.width
        sparse_depth = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        R = DepthAligner.qvec2rotmat(image.qvec)
        t = image.tvec

        valid_mask = image.point3D_ids != -1
        valid_point_ids = image.point3D_ids[valid_mask]
        valid_2d = image.xys[valid_mask]

        valid_points = []
        valid_errors = []
        valid_coords = []

        for i, point3D_id in enumerate(valid_point_ids):
            if point3D_id in points3D:
                point3D = points3D[point3D_id]
                valid_points.append(point3D.xyz)
                valid_errors.append(point3D.error)
                valid_coords.append(valid_2d[i])

        if len(valid_points) == 0:
            return sparse_depth, weight_map

        valid_points = np.array(valid_points)
        valid_errors = np.array(valid_errors)
        valid_coords = np.array(valid_coords)

        points_cam = (R @ valid_points.T).T + t
        depths = points_cam[:, 2]

        pixel_coords = valid_coords.astype(np.int32)

        valid_depth_mask = depths > 0
        pixel_coords = pixel_coords[valid_depth_mask]
        depths = depths[valid_depth_mask]
        valid_errors = valid_errors[valid_depth_mask]

        weights = 1.0 / (valid_errors + 1e-6)
        weights = weights / weights.max()

        valid_pixels = (
            (pixel_coords[:, 1] >= 0)
            & (pixel_coords[:, 1] < h)
            & (pixel_coords[:, 0] >= 0)
            & (pixel_coords[:, 0] < w)
        )
        pixel_coords = pixel_coords[valid_pixels]
        depths = depths[valid_pixels]
        weights = weights[valid_pixels]

        sparse_depth[pixel_coords[:, 1], pixel_coords[:, 0]] = depths
        weight_map[pixel_coords[:, 1], pixel_coords[:, 0]] = weights

        return sparse_depth, weight_map

    @staticmethod
    def align_depth_least_squares(
        dense_depth: np.ndarray,
        sparse_depth: np.ndarray,
        weight_map: np.ndarray,
    ) -> Tuple[float, float]:
        """Align dense monocular depth to sparse COLMAP depth via weighted LS."""
        valid_mask = sparse_depth > 0

        if valid_mask.sum() < 2:
            return 1.0, 0.0

        d_sparse = sparse_depth[valid_mask]
        d_dense = dense_depth[valid_mask]
        w = weight_map[valid_mask]
        w = w / w.sum()

        A = np.stack([d_dense, np.ones_like(d_dense)], axis=1)
        W = np.diag(w)
        b = d_sparse

        AtWA = A.T @ W @ A
        AtWb = A.T @ W @ b

        params = np.linalg.solve(AtWA, AtWb)
        scale, offset = params
        return scale, offset

    @staticmethod
    def align_depth_map(
        dense_depth: np.ndarray,
        image: Image,
        camera: CameraParams,
        points3D: Dict[int, Point3D],
    ) -> Tuple[np.ndarray, float, float, int]:
        """Align a dense depth map to COLMAP scale and return aligned depth."""
        sparse_depth, weight_map = DepthAligner.create_sparse_depth_map(
            image, camera, points3D
        )

        num_points = (sparse_depth > 0).sum()
        scale, offset = DepthAligner.align_depth_least_squares(
            dense_depth, sparse_depth, weight_map
        )
        aligned_depth = scale * dense_depth + offset
        return aligned_depth, scale, offset, num_points


# =============================================================================
# DEPTH MAP GENERATOR
# =============================================================================


class DepthMapGenerator:
    """Generate COLMAP-aligned depth maps and store them efficiently."""

    def __init__(
        self,
        colmap_sparse_path: str,
        image_dir: str,
        output_dir: str,
        model_type: str = "depth_anything_v2",
        device: str = "cuda",
        depth_dtype: str = "float16",
        compress: bool = True,
        da2_encoder: str = "vits",
        da2_ckpt_path: Optional[str] = None,
    ):
        self.colmap_path = Path(colmap_sparse_path)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.compress = compress
        self.depth_dtype = getattr(np, depth_dtype)

        print("Storage settings:")
        print(
            f"  - Precision: {depth_dtype} "
            f"({np.dtype(self.depth_dtype).itemsize} bytes/pixel)"
        )
        print(
            f"  - Compression: {'Enabled (.npz)' if compress else 'Disabled (.npy)'}"
        )

        print("\nLoading COLMAP sparse reconstruction...")
        self.cameras = COLMAPReader.read_cameras_binary(self.colmap_path / "cameras.bin")
        self.images = COLMAPReader.read_images_binary(self.colmap_path / "images.bin")
        self.points3D = COLMAPReader.read_points3D_binary(self.colmap_path / "points3D.bin")
        print(
            f"✓ Loaded {len(self.cameras)} cameras, "
            f"{len(self.images)} images, {len(self.points3D)} 3D points"
        )

        print(f"\nLoading {model_type} depth estimation model...")
        self.depth_estimator = DepthEstimator(
            model_type=model_type,
            device=device,
            da2_encoder=da2_encoder,
            da2_ckpt_path=da2_ckpt_path,
        )
        print("✓ Model loaded")

    def _save_depth(self, depth: np.ndarray, filename: str) -> int:
        """Save depth map to disk and return file size in bytes."""
        depth = depth.astype(self.depth_dtype)

        if self.compress:
            output_path = self.output_dir / (filename + ".npz")
            np.savez_compressed(output_path, depth=depth)
        else:
            output_path = self.output_dir / (filename + ".npy")
            np.save(output_path, depth)

        return output_path.stat().st_size

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Load an image as RGB."""
        if not image_path.exists():
            return None
        bgr_image = cv2.imread(str(image_path))
        if bgr_image is None:
            return None
        return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    def process_image(
        self,
        image_name: str,
        save_visualization: bool = False,
        pbar: Optional[tqdm] = None,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """Process a single image and return (aligned_depth, file_size)."""
        image_data = None
        for img in self.images.values():
            if img.name == image_name:
                image_data = img
                break

        if image_data is None:
            if pbar:
                pbar.write(f"⚠ {image_name} not found in COLMAP images.bin")
            return None

        image_path = self.image_dir / image_name
        rgb_image = self._load_image(image_path)
        if rgb_image is None:
            if pbar:
                pbar.write(f"⚠ Failed to load {image_path}")
            return None

        camera = self.cameras[image_data.camera_id]
        dense_depth = self.depth_estimator.predict(rgb_image)

        aligned_depth, scale, offset, num_points = DepthAligner.align_depth_map(
            dense_depth, image_data, camera, self.points3D
        )

        depth_filename = Path(image_name).stem + "_depth"
        file_size = self._save_depth(aligned_depth, depth_filename)

        if pbar:
            file_size_mb = file_size / (1024 * 1024)
            pbar.set_postfix({"size": f"{file_size_mb:.1f}MB", "pts": num_points})

        return aligned_depth, file_size

    def process_all_images(self):
        """Process all images, with a progress bar and summary stats."""
        image_names = [img.name for img in self.images.values()]

        print(f"\n{'=' * 70}")
        print(f"Processing {len(image_names)} images...")
        print(f"{'=' * 70}\n")

        successful = 0
        total_size = 0

        with tqdm(total=len(image_names), desc="Generating depth maps", unit="img") as pbar:
            for image_name in image_names:
                result = self.process_image(image_name, False, pbar)
                if result is not None:
                    _, file_size = result
                    successful += 1
                    total_size += file_size
                pbar.update(1)

        avg_size_mb = total_size / successful / (1024 * 1024) if successful > 0 else 0
        total_size_gb = total_size / (1024**3)

        print(f"\n{'=' * 70}")
        print(f"✓ Completed: {successful}/{len(image_names)} images")
        print("\nStorage Statistics:")
        print(f"  - Average file size: {avg_size_mb:.2f} MB")
        print(f"  - Total storage: {total_size_gb:.2f} GB")
        print(f"{'=' * 70}\n")
        print(f"Output: {self.output_dir.absolute()}")


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate COLMAP-aligned depth maps using Depth-Anything-V2 / ZoeDepth / MiDaS."
    )

    parser.add_argument("--colmap_sparse_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--model_type",
        type=str,
        default="depth_anything_v2",
        choices=["depth_anything_v2", "depth_anything_v2_hf", "zoedepth", "midas"],
        help=(
            "Which depth model backend to use:\n"
            "  depth_anything_v2      → official repo (local .pth checkpoint)\n"
            "  depth_anything_v2_hf   → HuggingFace Transformers backend\n"
            "  zoedepth               → ZoeDepth via torch.hub\n"
            "  midas                  → MiDaS via torch.hub"
        ),
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--depth_dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64"],
        help="Depth storage precision.",
    )
    parser.add_argument(
        "--no_compress",
        action="store_true",
        help="Disable .npz compression (use raw .npy instead).",
    )

    # Depth-Anything-V2 specific options
    parser.add_argument(
        "--da2-encoder",
        type=str,
        default="vits",
        choices=["vits", "vitb", "vitl"],
        help="Encoder size for DepthAnythingV2 official backend.",
    )
    parser.add_argument(
        "--da2-ckpt-path",
        type=str,
        default=None,
        help=(
            "Optional path to a local DepthAnythingV2 checkpoint "
            "(depth_anything_v2_*.pth) or a local HF model directory "
            "(when using depth_anything_v2_hf)."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available. Falling back to CPU.")
        args.device = "cpu"

    generator = DepthMapGenerator(
        colmap_sparse_path=args.colmap_sparse_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        device=args.device,
        depth_dtype=args.depth_dtype,
        compress=not args.no_compress,
        da2_encoder=args.da2_encoder,
        da2_ckpt_path=args.da2_ckpt_path,
    )
    generator.process_all_images()


if __name__ == "__main__":
    main()
