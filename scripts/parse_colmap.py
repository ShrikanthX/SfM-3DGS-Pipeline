from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
#!/usr/bin/env python3
"""
Enhanced COLMAP sparse reconstruction analysis with SfM quality metrics
and 3D Gaussian Splatting initialization diagnostics.
"""

import os
import sys
import struct
import collections
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass, asdict

# Define named tuples for COLMAP data structures
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

# COLMAP camera models
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

CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """Read cameras.bin file."""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=CAMERA_MODEL_IDS[model_id].model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
    return cameras


def read_images_binary(path_to_model_file):
    """Read images.bin file."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            ) if num_points2D > 0 else np.array([]).reshape(0, 2)
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            
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


def read_points3D_binary(path_to_model_file):
    """Read points3D.bin file."""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix."""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def compute_sfm_quality_metrics(cameras, images, points3D):
    """Compute comprehensive SfM quality and 3DGS-specific metrics."""
    metrics = {}
    
    # ============ RECONSTRUCTION OVERVIEW ============
    metrics["num_cameras"] = len(cameras)
    metrics["num_images"] = len(images)
    metrics["num_points3D"] = len(points3D)
    
    # ============ IMAGE-LEVEL METRICS ============
    observations_per_image = [len(img.point3D_ids) for img in images.values()]
    valid_observations = [
        np.sum(img.point3D_ids != -1) for img in images.values()
    ]
    
    metrics["images_with_valid_points"] = sum(
        1 for v in valid_observations if v > 0
    )
    metrics["mean_observations_per_image"] = np.mean(observations_per_image)
    metrics["median_observations_per_image"] = np.median(observations_per_image)
    metrics["min_observations_per_image"] = np.min(observations_per_image)
    metrics["max_observations_per_image"] = np.max(observations_per_image)
    metrics["std_observations_per_image"] = np.std(observations_per_image)
    
    # ============ 3D POINT METRICS ============
    errors = np.array([pt.error for pt in points3D.values()])
    track_lengths = np.array([len(pt.image_ids) for pt in points3D.values()])
    
    metrics["mean_reprojection_error"] = float(np.mean(errors))
    metrics["median_reprojection_error"] = float(np.median(errors))
    metrics["std_reprojection_error"] = float(np.std(errors))
    metrics["min_reprojection_error"] = float(np.min(errors))
    metrics["max_reprojection_error"] = float(np.max(errors))
    
    metrics["mean_track_length"] = float(np.mean(track_lengths))
    metrics["median_track_length"] = float(np.median(track_lengths))
    metrics["std_track_length"] = float(np.std(track_lengths))
    metrics["min_track_length"] = int(np.min(track_lengths))
    metrics["max_track_length"] = int(np.max(track_lengths))
    
    # ============ COVERAGE & CONNECTIVITY METRICS ============
    # Points visibility distribution
    metrics["points_visible_in_2_images"] = int(np.sum(track_lengths == 2))
    metrics["points_visible_in_3_plus_images"] = int(np.sum(track_lengths >= 3))
    metrics["points_visible_in_5_plus_images"] = int(np.sum(track_lengths >= 5))
    metrics["points_visible_in_10_plus_images"] = int(np.sum(track_lengths >= 10))
    
    # Reconstruction completeness
    metrics["point_visibility_ratio"] = float(
        np.sum(track_lengths >= 3) / len(points3D) * 100
        if len(points3D) > 0 else 0
    )
    
    # ============ CAMERA NETWORK QUALITY ============
    # Compute camera baseline statistics
    camera_positions = np.array([img.tvec for img in images.values()])
    if len(camera_positions) > 1:
        pairwise_distances = [
            np.linalg.norm(camera_positions[i] - camera_positions[j])
            for i in range(len(camera_positions))
            for j in range(i + 1, min(i + 5, len(camera_positions)))  # Compare with next 4
        ]
        metrics["mean_baseline"] = float(np.mean(pairwise_distances))
        metrics["median_baseline"] = float(np.median(pairwise_distances))
        metrics["max_baseline"] = float(np.max(pairwise_distances))
        metrics["min_baseline"] = float(np.min(pairwise_distances))
    
    # Scene scale estimation
    if len(points3D) > 0:
        point_positions = np.array([pt.xyz for pt in points3D.values()])
        scene_aabb = np.array([
            np.min(point_positions, axis=0),
            np.max(point_positions, axis=0)
        ])
        scene_extent = np.linalg.norm(scene_aabb[1] - scene_aabb[0])
        metrics["scene_extent"] = float(scene_extent)
        metrics["point_cloud_mean_x"] = float(np.mean(point_positions[:, 0]))
        metrics["point_cloud_mean_y"] = float(np.mean(point_positions[:, 1]))
        metrics["point_cloud_mean_z"] = float(np.mean(point_positions[:, 2]))
    
    # ============ CAMERA PARAMETERS ============
    camera_models_dict = {}
    for cam_id, cam in cameras.items():
        if cam.model not in camera_models_dict:
            camera_models_dict[cam.model] = {
                "count": 0,
                "cameras": []
            }
        camera_models_dict[cam.model]["count"] += 1
        camera_models_dict[cam.model]["cameras"].append({
            "camera_id": cam.id,
            "width": int(cam.width),
            "height": int(cam.height),
            "params": cam.params.tolist()
        })
    
    metrics["camera_models"] = camera_models_dict
    
    # ============ 3DGS INITIALIZATION QUALITY ============
    # Estimate initialization quality for 3DGS
    
    # Point density relative to scene size
    if metrics.get("scene_extent", 0) > 0:
        metrics["point_density"] = float(
            len(points3D) / (metrics["scene_extent"] ** 3)
        )
    
    # Mean observation count per point
    total_observations = sum(len(pt.image_ids) for pt in points3D.values())
    metrics["observations_per_point"] = (
        float(total_observations / len(points3D))
        if len(points3D) > 0 else 0
    )
    
    # Coverage uniformity (lower std = more uniform)
    metrics["observation_distribution_uniformity"] = 1.0 / (
        1.0 + metrics["std_observations_per_image"]
    ) if metrics["std_observations_per_image"] > 0 else 1.0
    
    # Overall initialization quality score (0-100)
    quality_factors = [
        min(100, metrics["mean_observations_per_image"] / 10.0 * 100),  # obs count
        min(100, metrics["mean_track_length"] / 10.0 * 100),  # track length
        100 - min(100, metrics["mean_reprojection_error"] * 10),  # reprojection error
        metrics["observation_distribution_uniformity"] * 100,  # uniformity
    ]
    metrics["initialization_quality_score"] = float(np.mean(quality_factors))
    
    return metrics


def get_quality_assessment(metrics):
    """Generate qualitative assessment of reconstruction quality."""
    assessment = {}
    
    # Observation coverage assessment
    if metrics["mean_observations_per_image"] > 2000:
        assessment["coverage"] = ("Excellent", "✅")
    elif metrics["mean_observations_per_image"] > 1000:
        assessment["coverage"] = ("Good", "✅")
    elif metrics["mean_observations_per_image"] > 300:
        assessment["coverage"] = ("Adequate", "⚠️")
    else:
        assessment["coverage"] = ("Poor", "❌")
    
    # Track length assessment
    if metrics["mean_track_length"] > 10:
        assessment["connectivity"] = ("Excellent", "✅")
    elif metrics["mean_track_length"] > 5:
        assessment["connectivity"] = ("Good", "✅")
    elif metrics["mean_track_length"] > 3:
        assessment["connectivity"] = ("Adequate", "⚠️")
    else:
        assessment["connectivity"] = ("Poor", "❌")
    
    # Reprojection error assessment
    if metrics["mean_reprojection_error"] < 1.0:
        assessment["accuracy"] = ("Excellent", "✅")
    elif metrics["mean_reprojection_error"] < 2.0:
        assessment["accuracy"] = ("Good", "✅")
    elif metrics["mean_reprojection_error"] < 3.0:
        assessment["accuracy"] = ("Acceptable", "⚠️")
    else:
        assessment["accuracy"] = ("Poor", "❌")
    
    # 3D-GS readiness assessment
    quality_score = metrics["initialization_quality_score"]
    if quality_score > 80:
        assessment["3dgs_readiness"] = ("Excellent", "✅")
    elif quality_score > 60:
        assessment["3dgs_readiness"] = ("Good", "✅")
    elif quality_score > 40:
        assessment["3dgs_readiness"] = ("Fair", "⚠️")
    else:
        assessment["3dgs_readiness"] = ("Poor", "❌")
    
    return assessment


def print_sfm_quality_report(metrics, assessment):
    """Print comprehensive SfM quality report."""
    print("\n" + "=" * 90)
    print("COLMAP SPARSE RECONSTRUCTION - SFM QUALITY & 3DGS READINESS REPORT")
    print("=" * 90)
    
    # Overview
    print(f"\n📊 RECONSTRUCTION OVERVIEW:")
    print(f"  Cameras: {metrics['num_cameras']}")
    print(f"  Images: {metrics['num_images']}")
    print(f"  3D Points: {metrics['num_points3D']}")
    print(f"  Scene Extent: {metrics.get('scene_extent', 0):.3f} units")
    
    # Quality Assessments
    print(f"\n🎯 QUALITY ASSESSMENT:")
    for category, (level, emoji) in assessment.items():
        print(f"  {category.replace('_', ' ').title():.<30} {emoji} {level}")
    
    # Coverage Metrics
    print(f"\n🔍 COVERAGE METRICS:")
    print(f"  Mean observations/image: {metrics['mean_observations_per_image']:.0f}")
    print(f"  Median observations/image: {metrics['median_observations_per_image']:.0f}")
    print(f"  Std dev: {metrics['std_observations_per_image']:.0f}")
    print(f"  Range: {metrics['min_observations_per_image']:.0f} - {metrics['max_observations_per_image']:.0f}")
    
    # Connectivity Metrics
    print(f"\n🔗 CONNECTIVITY METRICS:")
    print(f"  Mean track length: {metrics['mean_track_length']:.2f} images")
    print(f"  Median track length: {metrics['median_track_length']:.0f} images")
    print(f"  Points in 3+ images: {metrics['points_visible_in_3_plus_images']:,} ({metrics['point_visibility_ratio']:.1f}%)")
    print(f"  Points in 10+ images: {metrics['points_visible_in_10_plus_images']:,}")
    print(f"  Max track length: {metrics['max_track_length']} images")
    
    # Accuracy Metrics
    print(f"\n📐 ACCURACY METRICS:")
    print(f"  Mean reprojection error: {metrics['mean_reprojection_error']:.4f} pixels")
    print(f"  Median reprojection error: {metrics['median_reprojection_error']:.4f} pixels")
    print(f"  Std dev: {metrics['std_reprojection_error']:.4f} pixels")
    print(f"  Range: {metrics['min_reprojection_error']:.4f} - {metrics['max_reprojection_error']:.4f} pixels")
    
    # Camera Network
    print(f"\n📷 CAMERA NETWORK:")
    if "mean_baseline" in metrics:
        print(f"  Mean baseline: {metrics['mean_baseline']:.4f} units")
        print(f"  Baseline range: {metrics['min_baseline']:.4f} - {metrics['max_baseline']:.4f} units")
    
    for model_name, model_info in metrics["camera_models"].items():
        print(f"  {model_name}: {model_info['count']} camera(s)")
        for cam in model_info['cameras']:
            print(f"    - Camera {cam['camera_id']}: {cam['width']}x{cam['height']}, params={cam['params']}")
    
    # 3D-GS Specific Metrics
    print(f"\n🌐 3D GAUSSIAN SPLATTING INITIALIZATION METRICS:")
    print(f"  Point density: {metrics.get('point_density', 0):.6f} points/unit³")
    print(f"  Mean observations/point: {metrics['observations_per_point']:.2f}")
    print(f"  Observation uniformity: {metrics['observation_distribution_uniformity']:.2f}/1.0")
    print(f"  Initialization quality score: {metrics['initialization_quality_score']:.1f}/100")
    
    print("\n" + "=" * 90)
    print("INTERPRETATION GUIDELINES:")
    print("  • Coverage: More observations per image = better feature tracking")
    print("  • Connectivity: Higher track length = more robust point reconstruction")
    print("  • Accuracy: Lower reprojection error = more accurate camera poses")
    print("  • 3DGS Score: 80+: Excellent, 60-79: Good, 40-59: Fair, <40: Poor")
    print("=" * 90 + "\n")


def save_detailed_json(metrics, output_path):
    """Save complete metrics to JSON file."""
    with open(output_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            elif isinstance(o, (np.float64, np.float32)):
                return float(o)
            elif isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Object of type {type(o)} is not JSON serializable")
        
        json.dump(metrics, f, indent=2, default=convert)
    print(f"✅ Detailed metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced COLMAP analysis with SfM quality metrics and 3DGS diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parse_colmap.py sparse/0
  python parse_colmap.py sparse/0 --detailed
  python parse_colmap.py sparse/0 --json metrics.json
        """
    )
    
    parser.add_argument(
        "sparse_path",
        type=str,
        help="Path to COLMAP sparse reconstruction folder"
    )
    
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Save metrics to JSON file"
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed analysis"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    sparse_path = Path(args.sparse_path)
    if not sparse_path.exists():
        print(f"❌ Error: Path does not exist: {sparse_path}")
        sys.exit(1)
    
    cameras_file = sparse_path / "cameras.bin"
    images_file = sparse_path / "images.bin"
    points3D_file = sparse_path / "points3D.bin"
    
    missing_files = []
    for file, name in [(cameras_file, "cameras.bin"), 
                       (images_file, "images.bin"), 
                       (points3D_file, "points3D.bin")]:
        if not file.exists():
            missing_files.append(name)
    
    if missing_files:
        print(f"❌ Error: Missing required files: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Read binary files
    print("📖 Reading COLMAP binary files...")
    try:
        cameras = read_cameras_binary(str(cameras_file))
        images = read_images_binary(str(images_file))
        points3D = read_points3D_binary(str(points3D_file))
        print("✅ Successfully loaded reconstruction data\n")
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        sys.exit(1)
    
    # Compute metrics
    print("📊 Computing quality metrics...")
    metrics = compute_sfm_quality_metrics(cameras, images, points3D)
    assessment = get_quality_assessment(metrics)
    
    # Print report
    print_sfm_quality_report(metrics, assessment)
    
    # Save JSON if requested
    if args.json:
        save_detailed_json(metrics, args.json)


if __name__ == "__main__":
    main()
