"""
This file is modified from https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py

Adds helpers to write COLMAP binary models from FastMap containers and
includes a projection-based writer that fills 2D observations in images.bin.

Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
All rights reserved.
"""

from loguru import logger
import typing
from typing import Dict, Union
import os
import collections
import numpy as np
import struct
import torch

# FastMap containers
from fastmap.container import ColmapModel
from fastmap.container import Images as ImagesContainer
from fastmap.container import Cameras as CamerasContainer
from fastmap.container import Points3D as Points3DContainer


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {cm.model_id: cm for cm in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {cm.model_name: cm for cm in CAMERA_MODELS}


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    if isinstance(data, (list, tuple)):
        b = struct.pack(endian_character + format_char_sequence, *data)
    else:
        b = struct.pack(endian_character + format_char_sequence, data)
    fid.write(b)


def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_binary(cameras: Dict[int, Camera], path: str):
    if os.path.exists(path):
        raise FileExistsError(f"File {path} already exists.")
    with open(path, "wb") as fid:
        num_cameras = len(cameras)
        write_next_bytes(fid, num_cameras, "Q")
        for camera_id in sorted(cameras.keys()):
            camera: Camera = cameras[camera_id]
            if camera.model not in CAMERA_MODEL_NAMES:
                raise ValueError(f"Unsupported camera model: {camera.model}")
            model_id = CAMERA_MODEL_NAMES[camera.model].model_id
            num_params = CAMERA_MODEL_NAMES[camera.model].num_params
            if np.asarray(camera.params).shape != (num_params,):
                raise ValueError(
                    f"{camera.model} expects {num_params} params, got {np.asarray(camera.params).shape}"
                )
            write_next_bytes(fid, [camera.id, model_id, camera.width, camera.height], "iiQQ")
            write_next_bytes(fid, np.asarray(camera.params, np.float64).tolist(), "d" * num_params)


def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            # read name (null-terminated)
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D) if num_points2D > 0 else ()
            if num_points2D > 0:
                xys = np.column_stack(
                    [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            else:
                xys = np.zeros((0, 2), dtype=np.float64)
                point3D_ids = np.zeros((0,), dtype=np.int64)
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_binary(images: Dict[int, Image], path: str):
    """Write images (with poses and optional 2D observations) to a binary file."""
    if os.path.exists(path):
        raise FileExistsError(f"File {path} already exists.")
    with open(path, "wb") as fid:
        num_reg_images = len(images)
        write_next_bytes(fid, num_reg_images, "Q")
        for image_id in sorted(images.keys()):
            image: Image = images[image_id]
            props = [image.id] + list(np.asarray(image.qvec, np.float64)) + list(np.asarray(image.tvec, np.float64)) + [int(image.camera_id)]
            write_next_bytes(fid, props, "idddddddi")
            binary_image_name = image.name.encode("utf-8")
            write_next_bytes(fid, binary_image_name + b"\x00", f"{len(binary_image_name) + 1}s")

            # 2D obs
            if image.xys is None or image.point3D_ids is None:
                num_points2D = 0
                write_next_bytes(fid, num_points2D, "Q")
                continue

            xys = np.asarray(image.xys, dtype=np.float64)
            pids = np.asarray(image.point3D_ids, dtype=np.int64)
            assert xys.ndim == 2 and xys.shape[1] == 2
            assert pids.ndim == 1 and pids.shape[0] == xys.shape[0]

            num_points2D = xys.shape[0]
            write_next_bytes(fid, num_points2D, "Q")
            if num_points2D > 0:
                flat = []
                for (x, y), pid in zip(xys, pids):
                    flat.extend([float(x), float(y), int(pid)])
                write_next_bytes(fid, flat, "ddq" * num_points2D)


def read_points3D_text(path):
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            prop = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = prop[0]
            xyz = np.array(prop[1:4])
            rgb = np.array(prop[4:7])
            error = float(prop[7])
            track_length = read_next_bytes(fid, 8, "Q")[0]
            if track_length > 0:
                track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
                image_ids = np.array(tuple(map(int, track_elems[0::2])))
                point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            else:
                image_ids = np.zeros((0,), dtype=np.int32)
                point2D_idxs = np.zeros((0,), dtype=np.int32)
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def write_points3D_binary(points3D: Dict[int, Point3D], path: str):
    if os.path.exists(path):
        raise FileExistsError(f"File {path} already exists.")
    with open(path, "wb") as fid:
        num_points = len(points3D)
        write_next_bytes(fid, num_points, "Q")
        for point3D_id in sorted(points3D.keys()):
            p: Point3D = points3D[point3D_id]
            prop = [p.id] + p.xyz.astype(np.float64).tolist() + p.rgb.astype(np.uint8).tolist() + [float(p.error)]
            write_next_bytes(fid, prop, "QdddBBBd")
            track_length = len(p.image_ids)
            write_next_bytes(fid, track_length, "Q")
            if track_length > 0:
                elems = np.stack([p.image_ids.astype(np.int32), p.point2D_idxs.astype(np.int32)], axis=-1).reshape(-1).tolist()
                write_next_bytes(fid, elems, "ii" * track_length)


def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        logger.info(f"Detected model format: '{ext}'")
        return True
    return False


def qvec2rotmat(qvec):
    return np.array(
        [
            [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
            [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
            [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)

    # 1) guard against NaNs / Infs
    if not np.all(np.isfinite(R)):
        # fall back to identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # 2) project to SO(3) to remove numerical drift
    # R ~= U V^T
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        # fix improper rotation
        U[:, -1] *= -1
        R = U @ Vt

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    # ensure positive scalar part, to match COLMAP's convention
    if q[0] < 0:
        q = -q
    return q


def read_model(model_path: str, device: Union[torch.device, str] = "cpu", ext: str = ""):
    if ext == "":
        if detect_model_format(model_path, ".bin"):
            ext = ".bin"
        elif detect_model_format(model_path, ".txt"):
            ext = ".txt"
        else:
            raise Exception(f"Model not found in {model_path}")

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(model_path, "cameras" + ext))
        images = read_images_text(os.path.join(model_path, "images" + ext))
        points3D = read_points3D_text(os.path.join(model_path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(model_path, "cameras" + ext))
        images = read_images_binary(os.path.join(model_path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(model_path, "points3D") + ext)

    num_images = len(images)
    num_points3d = len(points3D)

    names = [""] * num_images
    rotation = torch.nan + torch.zeros(num_images, 3, 3, device=device, dtype=torch.float32)
    translation = torch.nan + torch.zeros(num_images, 3, device=device, dtype=torch.float32)
    focal = torch.nan + torch.zeros(num_images, device=device, dtype=torch.float32)
    k1 = torch.nan + torch.zeros(num_images, device=device, dtype=torch.float32)
    xyz = torch.nan + torch.zeros(num_points3d, 3, device=device, dtype=torch.float32)
    rgb = torch.zeros(num_points3d, 3, device=device, dtype=torch.uint8)

    for i, image_id in enumerate(sorted(images.keys())):
        names[i] = images[image_id].name
        rotation[i] = torch.from_numpy(qvec2rotmat(images[image_id].qvec)).to(rotation)
        translation[i] = torch.from_numpy(images[image_id].tvec).to(translation)

    for i, image_id in enumerate(sorted(images.keys())):
        cam = cameras[images[image_id].camera_id]
        if cam.model == "SIMPLE_PINHOLE":
            focal[i] = cam.params[0]
        elif cam.model == "PINHOLE":
            focal[i] = (cam.params[0] + cam.params[1]) / 2.0
        elif cam.model == "SIMPLE_RADIAL":
            focal[i] = cam.params[0]
            k1[i] = cam.params[-1]
        else:
            raise ValueError(f"Unsupported camera model: {cam.model}")

    for i, point3D_id in enumerate(sorted(points3D.keys())):
        p = points3D[point3D_id]
        xyz[i] = torch.from_numpy(p.xyz).to(xyz)
        rgb[i] = torch.from_numpy(p.rgb).to(rgb)

    assert not torch.any(torch.isnan(rotation))
    assert not torch.any(torch.isnan(translation))
    assert not torch.any(torch.isnan(focal))
    assert not torch.any(torch.isnan(xyz))
    assert len(names) == num_images

    model = ColmapModel(
        names=names,
        rotation=rotation,
        translation=translation,
        focal=focal,
        k1=k1,
        points3d=xyz,
        rgb=rgb,
    )
    return model


# ----------------------------
# Helpers for projection
# ----------------------------
def _project_points_xyz_to_pixels(
    xyz_w: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    f: float,
    cx: float,
    cy: float,
    k1: float = 0.0,
):
    """
    xyz_w: (N,3) world points
    R_w2c: (3,3)
    t_w2c: (3,)
    Returns: (N,2) pixel coords using SIMPLE_PINHOLE/SIMPLE_RADIAL (k1)
    """
    xyz_c = (R_w2c @ xyz_w.T + t_w2c[:, None]).T  # (N,3)
    z = xyz_c[:, 2]
    valid = z > 1e-8
    uv = np.zeros((xyz_w.shape[0], 2), dtype=np.float64)
    if not np.any(valid):
        return uv, valid
    x = xyz_c[valid, 0] / z[valid]
    y = xyz_c[valid, 1] / z[valid]
    if abs(k1) > 0.0:
        r2 = x * x + y * y
        scale = 1.0 + k1 * r2
        x = x * scale
        y = y * scale
    uv_valid = np.stack([f * x + cx, f * y + cy], axis=-1)  # (Nv,2)
    uv[valid] = uv_valid
    return uv, valid


def write_model(
    save_dir: str,
    images: ImagesContainer,
    cameras: CamerasContainer,
    points3d: Points3DContainer,
    R_w2c: torch.Tensor,
    t_w2c: torch.Tensor,
):
    """
    Write results to COLMAP-format binaries: cameras.bin, images.bin, points3D.bin.

    IMPORTANT:
    - Uses cameras.camera_idx (per-image camera indices) instead of a non-existent
      images.image_camera_ids.
    - Populates 2D observations in images.bin by projecting each triangulated 3D point
      into the images where it was observed (according to the point tracks). This
      avoids empty point3D_ids arrays.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving the results to {save_dir}")

    # --------------------
    # Build cameras dict
    # --------------------
    cameras_dict: Dict[int, Camera] = {}
    tol = 1e-12

    # camera i is 0-based in our containers; COLMAP uses 1-based IDs
    for i in range(cameras.num_cameras):
        camera_id = i + 1

        # which images use this camera?
        if hasattr(cameras, "camera_idx"):
            image_mask = (cameras.camera_idx == i)  # (num_images,)
        else:
            raise AttributeError("Cameras container must have 'camera_idx' (per-image).")

        if not hasattr(images, "widths") or not hasattr(images, "heights"):
            raise AttributeError("Images container must have 'widths' and 'heights'.")

        # derive width/height from first image using this camera
        sel_w = images.widths[image_mask]
        sel_h = images.heights[image_mask]
        assert sel_w.numel() > 0 and sel_h.numel() > 0, f"No images found for camera index {i}"
        width = int(sel_w.reshape(-1)[0].item())
        height = int(sel_h.reshape(-1)[0].item())

        f = float(cameras.focal[i].item())
        cx = float(cameras.cx[i].item())
        cy = float(cameras.cy[i].item())
        k1 = float(getattr(cameras, "k1", torch.tensor([0.0], device=cameras.focal.device))[i].item())

        if abs(k1) <= tol:
            model_name = "SIMPLE_PINHOLE"
            params = np.array([f, cx, cy], dtype=np.float64)
        else:
            model_name = "SIMPLE_RADIAL"
            params = np.array([f, cx, cy, k1], dtype=np.float64)

        cameras_dict[camera_id] = Camera(
            id=camera_id, model=model_name, width=width, height=height, params=params
        )

    # --------------------
    # Precompute per-image pose & intrinsics
    # --------------------
    num_images = images.num_images
    assert R_w2c.shape[:2] == (num_images, 3)
    assert t_w2c.shape[1] == 3

    # per image -> camera_id (1-based)
    if hasattr(cameras, "camera_idx"):
        per_image_cam_id = (cameras.camera_idx.detach().cpu().numpy().astype(np.int64) + 1)
    else:
        raise AttributeError("Cameras container must have 'camera_idx'.")

    f_all = cameras.focal.detach().cpu().numpy().astype(np.float64)
    cx_all = cameras.cx.detach().cpu().numpy().astype(np.float64)
    cy_all = cameras.cy.detach().cpu().numpy().astype(np.float64)
    if hasattr(cameras, "k1"):
        k1_all = cameras.k1.detach().cpu().numpy().astype(np.float64)
    else:
        k1_all = np.zeros_like(f_all)

    # --------------------
    # Build images dict with 2D obs from tracks/points
    # --------------------
    # Allocate lists to accumulate per-image observations.
    per_image_xys: typing.List[list] = [list() for _ in range(num_images)]
    per_image_pids: typing.List[list] = [list() for _ in range(num_images)]

    # For each 3D point, project it into every image where it has a track hit.
    # points3d.track_image_idx is a list of lists of image indices (0-based).
    xyz_all = points3d.xyz.detach().cpu().numpy().astype(np.float64)
    for pid0 in range(points3d.num_points):
        pid = pid0 + 1  # COLMAP 1-based point id
        xyz = xyz_all[pid0 : pid0 + 1]  # (1,3)

        img_list = points3d.track_image_idx[pid0]
        if len(img_list) == 0:
            continue

        for img_idx in img_list:
            img_idx = int(img_idx)
            # pose
            R = R_w2c[img_idx].detach().cpu().numpy().astype(np.float64)
            t = t_w2c[img_idx].detach().cpu().numpy().astype(np.float64)
            # intrinsics from the camera used by this image
            cam_id_1b = per_image_cam_id[img_idx]
            cam0 = cameras_dict[int(cam_id_1b)]
            if cam0.model == "SIMPLE_PINHOLE":
                f, cx, cy = cam0.params.tolist()
                k1 = 0.0
            else:
                f, cx, cy, k1 = cam0.params.tolist()

            uv, valid = _project_points_xyz_to_pixels(xyz, R, t, f, cx, cy, k1)
            if not bool(valid[0]):
                continue  # behind camera

            u, v = uv[0]
            # Optional: you can clamp to image bounds or skip out-of-bounds; we keep all
            per_image_xys[img_idx].append([u, v])
            per_image_pids[img_idx].append(pid)

    # Now assemble Image entries
    images_dict: Dict[int, Image] = {}
    for i in range(num_images):
        image_id = i + 1
        image_name = images.names[i]
        camera_id = int(per_image_cam_id[i])

        # pose
        R = R_w2c[i].detach().cpu().numpy().astype(np.float64)
        t = t_w2c[i].detach().cpu().numpy().astype(np.float64)
        qvec = rotmat2qvec(R)

        xys_i = np.asarray(per_image_xys[i], dtype=np.float64) if len(per_image_xys[i]) > 0 else np.zeros((0, 2), dtype=np.float64)
        pids_i = np.asarray(per_image_pids[i], dtype=np.int64) if len(per_image_pids[i]) > 0 else np.zeros((0,), dtype=np.int64)

        images_dict[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=t,
            camera_id=camera_id,
            name=image_name,
            xys=xys_i,
            point3D_ids=pids_i,
        )

    # --------------------
    # Build points3D dict
    # --------------------
    points3d_dict: Dict[int, Point3D] = {}
    for i in range(points3d.num_points):
        point3D_id = i + 1
        xyz = points3d.xyz[i].detach().cpu().numpy().astype(np.float64)
        rgb = points3d.rgb[i].detach().cpu().numpy().astype(np.uint8)
        error = float(points3d.error[i].detach().cpu().numpy()) if hasattr(points3d, "error") else 0.0

        # tracks in COLMAP use 1-based image IDs and 0-based keypoint indices
        img_ids = np.asarray(points3d.track_image_idx[i], dtype=np.int32) + 1
        kp_ids = np.asarray(points3d.track_keypoint_idx[i], dtype=np.int32)

        points3d_dict[point3D_id] = Point3D(
            id=point3D_id,
            xyz=xyz,
            rgb=rgb,
            error=error,
            image_ids=img_ids,
            point2D_idxs=kp_ids,
        )

    # --------------------
    # Write binaries
    # --------------------
    write_cameras_binary(cameras_dict, os.path.join(save_dir, "cameras.bin"))
    write_images_binary(images_dict, os.path.join(save_dir, "images.bin"))
    write_points3D_binary(points3d_dict, os.path.join(save_dir, "points3D.bin"))

    logger.info("Cameras are written to %s", os.path.join(save_dir, "cameras.bin"))
    logger.info("Images are written to %s", os.path.join(save_dir, "images.bin"))
    logger.info("3D points are written to %s", os.path.join(save_dir, "points3D.bin"))