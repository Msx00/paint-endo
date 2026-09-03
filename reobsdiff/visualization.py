"""Fixed paper/debug visualization panels (never loads E1 itself)."""

from pathlib import Path
import cv2
import numpy as np


def _save(path, image):
    value = np.asarray(image)
    if value.dtype != np.uint8:
        value = np.clip(value * 255, 0, 255).astype(np.uint8)
    if value.ndim == 3:
        value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), value)


def save_cache_visualization(output, arrays, prediction=None, e1_gt=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    hole = arrays["virtual_mask"].astype(bool)
    reobs = arrays["reobs_mask"].astype(bool)
    pred = arrays["virtual_warp"] if prediction is None else prediction
    final = arrays["virtual_warp"] * (~hole)[..., None] + pred * hole[..., None]
    panels = [("01_e2_anchor.png", arrays["anchor_rgb"]),
              ("02_virtual_warp.png", arrays["virtual_warp"]),
              ("03_geometry_hole_mask.png", hole),
              ("04_reciprocal_corruption.png", arrays["reciprocal_rgb"]),
              ("05_diffusion_prediction.png", pred),
              ("06_reobservation_rgb.png", arrays["reobs_rgb"]),
              ("07_reobservation_mask.png", reobs),
              ("08_reobservation_confidence.png", arrays["reobs_confidence"]),
              ("09_final_hard_composition.png", final)]
    if e1_gt is not None:
        panels.append(("10_e1_gt_EVALUATION_ONLY.png", e1_gt))
    for name, value in panels:
        _save(output / name, value)
    overlay = np.zeros((*hole.shape, 3), np.uint8)
    overlay[~hole] = (0, 255, 0)
    overlay[hole & reobs] = (255, 0, 0)
    overlay[hole & ~reobs] = (255, 255, 0)
    _save(output / "11_support_overlay.png", overlay)
