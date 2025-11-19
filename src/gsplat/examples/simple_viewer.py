import argparse
import math
import os
import time
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
import viser.transforms as vtf

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState

# Dataset parser used during training (for SfM initialization)
try:
    from datasets.colmap import Parser as ColmapParser

    _HAS_COLMAP_PARSER = True
except Exception:
    _HAS_COLMAP_PARSER = False


# -------------------------------------------------------------------------
# Minimal cfg.yml parser (avoid python/object YAML issues)
# -------------------------------------------------------------------------
def _parse_training_cfg_minimal(cfg_path: Path):
    """
    Minimal, robust parser for cfg.yml that only extracts the fields we need:

        data_dir: <str>
        data_factor: <int>
        normalize_world_space: <bool>
        test_every: <int>

    Ignores everything else (python/tuple, python/object tags, etc.).
    """
    cfg = {
        "data_dir": None,
        "data_factor": 1,
        "normalize_world_space": True,
        "test_every": 8,
    }

    if not cfg_path.exists():
        print(f"[SfM] No cfg.yml found at {cfg_path}")
        return cfg

    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if (s.startswith("'") and s.endswith("'")) or (
            s.startswith('"') and s.endswith('"')
        ):
            return s[1:-1]
        return s

    with open(cfg_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("data_dir:"):
                val = stripped.split(":", 1)[1].strip()
                cfg["data_dir"] = _strip_quotes(val)
            elif stripped.startswith("data_factor:"):
                val = stripped.split(":", 1)[1].strip()
                try:
                    cfg["data_factor"] = int(val)
                except ValueError:
                    pass
            elif stripped.startswith("normalize_world_space:"):
                val = stripped.split(":", 1)[1].strip().lower()
                if val in ["true", "yes", "on"]:
                    cfg["normalize_world_space"] = True
                elif val in ["false", "no", "off"]:
                    cfg["normalize_world_space"] = False
            elif stripped.startswith("test_every:"):
                val = stripped.split(":", 1)[1].strip()
                try:
                    cfg["test_every"] = int(val)
                except ValueError:
                    pass

    return cfg


# -------------------------------------------------------------------------
# SfM overlay (points + camera frustums) from Parser
# -------------------------------------------------------------------------
def add_sfm_to_viewer(
    server: viser.ViserServer,
    parser,
    data_dir: str,
    tab_group,
    max_points: int = 200_000,
) -> None:
    """
    Overlay the SfM point cloud + camera frustums used to initialize GSplat.

    Everything is taken from `parser` and is already in the SAME normalized
    coordinate frame as the Gaussians (parser.points, parser.camtoworlds, etc.).
    All SfM controls live in a dedicated tab within the provided tab_group.
    """
    # ----- Point cloud -----
    if not hasattr(parser, "points") or parser.points is None:
        print("[SfM] Parser has no 'points'; skipping SfM overlay.")
        return
    if not hasattr(parser, "points_rgb") or parser.points_rgb is None:
        print("[SfM] Parser has no 'points_rgb'; skipping SfM overlay.")
        return

    xyz_all = np.asarray(parser.points, dtype=np.float32)  # [N, 3]
    rgb_all = np.asarray(parser.points_rgb, dtype=np.float32) / 255.0  # [N, 3]
    n_points = xyz_all.shape[0]
    if n_points == 0:
        print("[SfM] Parser points array is empty; skipping SfM overlay.")
        return

    print(f"[SfM] Loaded {n_points} SfM points from training parser.")

    # ----- Cameras / poses -----
    camtoworlds = None
    n_cams = 0
    if hasattr(parser, "camtoworlds") and parser.camtoworlds is not None:
        camtoworlds = np.asarray(
            parser.camtoworlds, dtype=np.float32
        )  # [N_cam, 3x4 or 4x4]
        if camtoworlds.shape[-2:] == (3, 4):
            ones = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)[None, None, :]
            camtoworlds = np.concatenate(
                [camtoworlds, np.repeat(ones, len(camtoworlds), axis=0)], axis=1
            )
        n_cams = camtoworlds.shape[0]

    # ----- Intrinsics & image size (for frustum shape) -----
    K = None
    width = height = None
    if hasattr(parser, "Ks_dict") and parser.Ks_dict:
        K = np.asarray(list(parser.Ks_dict.values())[0], dtype=np.float32)
    if hasattr(parser, "imsize_dict") and parser.imsize_dict:
        width, height = list(parser.imsize_dict.values())[0]
    if K is not None and width is not None and height is not None:
        fy = K[1, 1]
        fov = 2.0 * np.arctan2(height / 2.0, fy)
        aspect = float(width) / float(height)
    else:
        fov = math.radians(60.0)
        aspect = 16.0 / 9.0

    # ----- Images directory for frustum textures (best-effort) -----
    images_dir = Path(data_dir) / "images"
    image_paths = []
    if images_dir.exists():
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        for ext in exts:
            image_paths.extend(sorted(images_dir.glob(f"*{ext}")))
    else:
        print(f"[SfM] Images dir not found: {images_dir}")

    # ------------------------------------------------------------------
    # Put ALL SfM controls inside their own tab (separate tab group)
    # ------------------------------------------------------------------
    with tab_group.add_tab("SfM / Cameras", viser.Icon.CAMERA):
        gui_show_points = server.gui.add_checkbox(
            "Show SfM point cloud", initial_value=True
        )
        gui_show_cameras = server.gui.add_checkbox(
            "Show cameras & frustums", initial_value=True
        )

        gui_point_size = server.gui.add_slider(
            "SfM point size",
            min=0.001,
            max=0.05,
            step=0.001,
            initial_value=0.008,
        )
        gui_max_points = server.gui.add_slider(
            "Max SfM points",
            min=1000,
            max=max(1000, n_points),
            step=max(1, n_points // 200),
            initial_value=min(30_000, n_points),
        )
        gui_max_frames = server.gui.add_slider(
            "Max camera frames",
            min=0,
            max=n_cams,
            step=1,
            initial_value=n_cams,  # show all by default
        )
        gui_cam_scale = server.gui.add_slider(
            "Camera frustum scale",
            min=0.01,
            max=0.3,
            step=0.01,
            initial_value=0.05,
        )

        gui_override_color = server.gui.add_checkbox(
            "Override SfM color", initial_value=False
        )
        gui_color_r = server.gui.add_slider(
            "SfM color R", min=0.0, max=1.0, step=0.01, initial_value=0.0
        )
        gui_color_g = server.gui.add_slider(
            "SfM color G", min=0.0, max=1.0, step=0.01, initial_value=1.0
        )
        gui_color_b = server.gui.add_slider(
            "SfM color B", min=0.0, max=1.0, step=0.01, initial_value=0.0
        )

        server.gui.add_markdown(
            f"**SfM summary**  \n"
            f"- Points: `{n_points}`  \n"
            f"- Cameras: `{n_cams}`"
        )

        # ----- Subsampling for points -----
        def compute_mask(max_pts: int) -> np.ndarray:
            max_pts = min(max_pts, n_points)
            if max_pts <= 0:
                return np.zeros((n_points,), dtype=bool)
            if max_pts >= n_points:
                return np.ones((n_points,), dtype=bool)
            idx = np.random.choice(n_points, size=max_pts, replace=False)
            mask_local = np.zeros((n_points,), dtype=bool)
            mask_local[idx] = True
            return mask_local

        mask = compute_mask(gui_max_points.value)

        # Point cloud
        point_cloud = server.scene.add_point_cloud(
            name="/sfm/pcd",
            points=xyz_all[mask],
            colors=rgb_all[mask],
            point_size=gui_point_size.value,
        )

        def _get_override_color_vec() -> np.ndarray:
            return np.array(
                [gui_color_r.value, gui_color_g.value, gui_color_b.value],
                dtype=np.float32,
            )

        def _update_point_colors():
            if gui_override_color.value:
                col = _get_override_color_vec()
                point_cloud.colors = np.broadcast_to(col[None, :], (mask.sum(), 3))
            else:
                point_cloud.colors = rgb_all[mask]

        # ----- Cameras & frustums -----
        frames = []
        frustums = []
        first_frame = None

        if camtoworlds is not None and n_cams > 0:
            for idx_cam in range(n_cams):
                T = camtoworlds[idx_cam]  # [4,4] camera-to-world
                R_wc = T[:3, :3]
                t_wc = T[:3, 3]
                se3 = vtf.SE3.from_rotation_and_translation(
                    vtf.SO3.from_matrix(R_wc), t_wc
                )

                frame = server.scene.add_frame(
                    f"/sfm/frame_{idx_cam}",
                    wxyz=se3.rotation().wxyz,
                    position=se3.translation(),
                    axes_length=0.05,
                    axes_radius=0.002,
                )
                frames.append(frame)
                if first_frame is None:
                    first_frame = frame

                # Load image if available
                image = None
                if idx_cam < len(image_paths):
                    try:
                        img = imageio.imread(image_paths[idx_cam])
                        img = img[::2, ::2]  # light downsample for UI perf
                        image = img
                    except Exception as e:
                        print(
                            f"[SfM] Failed to load image {image_paths[idx_cam]}: {e}"
                        )

                frustum = server.scene.add_camera_frustum(
                    f"/sfm/frame_{idx_cam}/frustum",
                    fov=fov,
                    aspect=aspect,
                    scale=gui_cam_scale.value,
                    image=image,
                )
                frustums.append(frustum)

                @frustum.on_click
                def _(_, frame=frame) -> None:
                    for client in server.get_clients().values():
                        client.camera.wxyz = frame.wxyz
                        client.camera.position = frame.position

        # Snap initial / new clients to first frame if available
        if first_frame is not None:
            for client in server.get_clients().values():
                client.camera.wxyz = first_frame.wxyz
                client.camera.position = first_frame.position

            @server.on_client_connect
            def _on_connect(
                new_client: viser.ClientHandle, frame=first_frame
            ) -> None:
                new_client.camera.wxyz = frame.wxyz
                new_client.camera.position = frame.position

        # ------------------------------------------------------------------
        # UI callbacks (still inside the SfM tab context)
        # ------------------------------------------------------------------
        @gui_point_size.on_update
        def _(_event) -> None:
            point_cloud.point_size = gui_point_size.value

        @gui_max_points.on_update
        def _(_event) -> None:
            nonlocal mask
            mask = compute_mask(gui_max_points.value)
            with server.atomic():
                point_cloud.points = xyz_all[mask]
                _update_point_colors()

        @gui_show_points.on_update
        def _(_event) -> None:
            point_cloud.visible = gui_show_points.value

        @gui_show_cameras.on_update
        def _(_event) -> None:
            max_frames = gui_max_frames.value
            for i, frame in enumerate(frames):
                visible = (i < max_frames) and gui_show_cameras.value
                frame.visible = visible
            for i, frustum in enumerate(frustums):
                visible = (i < max_frames) and gui_show_cameras.value
                frustum.visible = visible

        @gui_max_frames.on_update
        def _(_event) -> None:
            max_frames = gui_max_frames.value
            for i, frame in enumerate(frames):
                visible = (i < max_frames) and gui_show_cameras.value
                frame.visible = visible
            for i, frustum in enumerate(frustums):
                visible = (i < max_frames) and gui_show_cameras.value
                frustum.visible = visible

        @gui_cam_scale.on_update
        def _(_event) -> None:
            for frustum in frustums:
                frustum.scale = gui_cam_scale.value

        @gui_override_color.on_update
        def _(_event) -> None:
            _update_point_colors()

        @gui_color_r.on_update
        def _(_event) -> None:
            if gui_override_color.value:
                _update_point_colors()

        @gui_color_g.on_update
        def _(_event) -> None:
            if gui_override_color.value:
                _update_point_colors()

        @gui_color_b.on_update
        def _(_event) -> None:
            if gui_override_color.value:
                _update_point_colors()


# -------------------------------------------------------------------------
# Main GSplat viewer
# -------------------------------------------------------------------------
def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ckpt is None:
        # Demo mode with synthetic test data from gsplat._helper
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        # batched render
        for _ in tqdm.trange(1):
            render_colors, render_alphas, meta = rasterization(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                Ks,
                width,
                height,
                render_mode="RGB+D",
                packed=False,
                distributed=world_size > 1,
            )
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)
        render_colors.sum().backward()

        render_rgbs = render_colors[..., 0:3]
        render_depths = render_colors[..., 3:4]
        render_depths = render_depths / render_depths.max()

        # dump batch images
        os.makedirs(args.output_dir, exist_ok=True)
        canvas = (
            torch.cat(
                [
                    render_rgbs.reshape(C * height, width, 3),
                    render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                    render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                ],
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
        )
        imageio.imsave(
            f"{args.output_dir}/render_rank{world_rank}.png",
            (canvas * 255).astype(np.uint8),
        )
    else:
        # Viewer mode for trained checkpoints
        means_list, quats_list, scales_list, opacities_list, sh0_list, shN_list = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means_list.append(ckpt["means"])
            quats_list.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales_list.append(torch.exp(ckpt["scales"]))
            opacities_list.append(torch.sigmoid(ckpt["opacities"]))
            sh0_list.append(ckpt["sh0"])
            shN_list.append(ckpt["shN"])

        means = torch.cat(means_list, dim=0)
        quats = torch.cat(quats_list, dim=0)
        scales = torch.cat(scales_list, dim=0)
        opacities = torch.cat(opacities_list, dim=0)
        sh0 = torch.cat(sh0_list, dim=0)
        shN = torch.cat(shN_list, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        print("Number of Gaussians:", len(means))

    # ---------------------------------------------------------------------
    # Nerfview / Gsplat viewer integration
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat[None],
            K[None],
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
        )
        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            render_colors_local = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors_local.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            render_colors_local = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors_local.cpu().numpy()
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)

    # SfM tab group – separate from whatever nerfview/GsplatViewer creates
    if args.ckpt is not None and _HAS_COLMAP_PARSER:
        cfg_path = Path(args.output_dir) / "cfg.yml"
        cfg = _parse_training_cfg_minimal(cfg_path)

        data_dir = cfg["data_dir"]
        data_factor = cfg["data_factor"]
        normalize = cfg["normalize_world_space"]
        test_every = cfg["test_every"]

        print("[SfM] Parsed minimal training config for overlay:")
        print(f"      data_dir={data_dir}")
        print(f"      data_factor={data_factor}")
        print(f"      normalize_world_space={normalize}")
        print(f"      test_every={test_every}")

        if data_dir is not None:
            try:
                parser_obj = ColmapParser(
                    data_dir=data_dir,
                    factor=data_factor,
                    normalize=normalize,
                    test_every=test_every,
                )
                sfm_tab_group = server.gui.add_tab_group()
                add_sfm_to_viewer(
                    server,
                    parser_obj,
                    data_dir=data_dir,
                    tab_group=sfm_tab_group,
                )
                print(
                    "[SfM] SfM point cloud + cameras ready (aligned to GSplat)."
                )
            except Exception as exc:
                print(f"[SfM] Failed to overlay SfM from parser: {exc}")
        else:
            print("[SfM] data_dir missing in cfg.yml; skipping SfM overlay.")
    else:
        if not _HAS_COLMAP_PARSER:
            print("[SfM] datasets.colmap.Parser not available; cannot overlay SfM.")

    # GSplat viewer (uses its own rendering UI/tabs)
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )

    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    Example:
    CUDA_VISIBLE_DEVICES=0 python src/gsplat/examples/simple_viewer.py \\
        --ckpt ./data/garden/results/ckpts/ckpt_29999_rank0.pt \\
        --output_dir ./data/garden/results \\
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")

    # Kept for backwards compatibility with your old call signature.
    parser.add_argument(
        "--scene_root",
        type=str,
        default=None,
        help="(Unused) kept for compatibility.",
    )
    parser.add_argument(
        "--colmap_sparse",
        type=str,
        default="sparse/0",
        help="(Unused) kept for compatibility.",
    )

    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
