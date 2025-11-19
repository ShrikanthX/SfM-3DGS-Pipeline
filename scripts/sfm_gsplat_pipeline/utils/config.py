"""Config and preset utilities for the unified pipeline.

Moved from scripts/run_full_pipeline.py to keep the main pipeline
orchestrator smaller and easier to read.
"""

def analyze_dataset_and_recommend_preset(images_dir: Path) -> Tuple[str, dict]:
    """Analyze dataset characteristics and recommend optimal preset."""
    if not AUTO_PRESET_AVAILABLE:
        return "balanced", {"reason": "Default (auto-detection unavailable)"}

    image_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    images = []
    for ext in image_extensions:
        images.extend(images_dir.glob(f"*{ext}"))
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    num_images = len(images)

    resolutions = []
    for img_path in images[:min(10, len(images))]:
        try:
            with Image.open(img_path) as img:
                resolutions.append(img.width * img.height)
        except Exception:
            pass
    if not resolutions:
        raise RuntimeError("Could not read any images for analysis")

    avg_resolution = float(np.mean(resolutions))
    megapixels = avg_resolution / 1e6

    info = {"num_images": num_images, "megapixels": megapixels}

    if num_images < 50:
        preset = "fast"; reason = "Small dataset (< 50 images)"
    elif num_images > 300:
        preset = "balanced"; reason = "Large dataset (> 300 images)"
    elif megapixels > 20:
        preset = "balanced"; reason = "High resolution images (> 20 MP)"
    elif megapixels < 2:
        preset = "quality"; reason = "Low resolution images (< 2 MP) - can afford full quality"
    else:
        preset = "balanced"; reason = "Standard dataset"

    info["reason"] = reason
    info["preset"] = preset
    return preset, info




def extend_cmd_from_config(cmd: list[str], prefix: str, cfg: Dict[str, Any]) -> None:
    """
    Convert a config dict into CLI flags and extend cmd.

    - prefix=""          → --batch-size, --max-steps
    - prefix="strategy." → --strategy.cap-max, etc.

    Types:
      - bool:   True → add flag, False → skip
      - list:   repeated flags
      - scalar: flag + value
    """
    for key, val in cfg.items():
        if val is None:
            continue

        flag = "--" + (prefix + key).replace("_", "-")

        # bool flags
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
            continue

        # list / tuple -> repeat
        if isinstance(val, (list, tuple)):
            for item in val:
                cmd.extend([flag, str(item)])
            continue

        # everything else
        cmd.extend([flag, str(val)])




def apply_overrides(target: Dict[str, Any], overrides: list[str]) -> None:
    """
    Apply key=value overrides into target dict, supporting dotted paths:

        strategy.cap_max=2000000
        depth_lambda=0.05

    Values are parsed with ast.literal_eval when possible.
    """
    for item in overrides:
        if "=" not in item:
            console.print(f"[yellow]⚠️ Ignoring override without '=': {item}[/yellow]")
            continue
        key, raw_val = item.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if not key:
            continue

        # parse value
        try:
            val = ast.literal_eval(raw_val)
        except Exception:
            # fallback to string
            val = raw_val

        # dotted key -> nested dict
        parts = key.split(".")
        cur = target
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val

# =============================================================================
# PIPELINE CLASS
# =============================================================================



