"""Robust stereo-temporal observations grounded on the nearest surface."""

import numpy as np
from reobsdiff.geometry.warp import warp_to_pose


def build_reobservation(anchor_valid, observations, K_virtual, T_virtual, config):
    projections = []
    for obs in observations:
        warped = warp_to_pose(obs["rgb"], obs["depth"], obs["K"], obs["T"],
                              K_virtual, T_virtual, obs.get("valid_depth_mask"),
                              obs.get("depth_confidence"))
        temporal = np.exp(-abs(float(obs.get("delta_t", 0))) / max(config.temporal_tau, 1e-6))
        view = float(obs.get("view_weight", 1.0))
        projections.append((warped, temporal * view, obs.get("view", "left")))
    shape = np.asarray(anchor_valid).shape
    if not projections:
        return {"reobs_rgb": np.zeros((*shape, 3), np.float32),
                "reobs_mask": np.zeros(shape, bool), "reobs_confidence": np.zeros(shape, np.float32),
                "observation_count": np.zeros(shape, np.uint16), "stereo_agreement": 0.0}
    depths = np.stack([p[0]["warped_depth"] for p in projections])
    valids = np.stack([p[0]["valid_mask"] for p in projections])
    safe_depths = np.where(valids, depths, np.inf)
    front = safe_depths.min(axis=0)
    consistent = valids.copy()
    if config.depth_consistency_mm > 0:
        consistent &= np.abs(depths - front) < config.depth_consistency_mm
    if config.depth_consistency_relative > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            relative_error = np.abs(depths - front) / np.maximum(front, 1e-6)
        consistent &= np.isfinite(relative_error) & (relative_error < config.depth_consistency_relative)
    images = np.stack([p[0]["warped_rgb"] for p in projections])
    base_weights = np.stack([p[0]["confidence"] * p[1] for p in projections])
    weights = base_weights * consistent
    initial_sum = weights.sum(axis=0)
    initial_rgb = (images * weights[..., None]).sum(axis=0) / np.maximum(initial_sum[..., None], 1e-8)
    error = np.mean(np.abs(images - initial_rgb[None]), axis=-1)
    photometric = np.clip(1.0 - error / max(config.reobs_max_photometric_error, 1e-6), 0, 1)
    weights *= photometric
    weight_sum = weights.sum(axis=0)
    rgb = (images * weights[..., None]).sum(axis=0) / np.maximum(weight_sum[..., None], 1e-8)
    hole = ~np.asarray(anchor_valid, bool)
    mask = hole & (weight_sum > 1e-8)
    count = (weights > 0).sum(axis=0).astype(np.uint16)
    confidence = np.clip(weight_sum / np.maximum(count, 1), 0, 1).astype(np.float32)
    if not config.use_reobs_confidence:
        confidence[mask] = 1.0
    left = np.any(np.stack([consistent[i] for i, p in enumerate(projections) if p[2] == "left"]), axis=0) if any(p[2] == "left" for p in projections) else np.zeros(shape, bool)
    right = np.any(np.stack([consistent[i] for i, p in enumerate(projections) if p[2] == "right"]), axis=0) if any(p[2] == "right" for p in projections) else np.zeros(shape, bool)
    stereo_agreement = float((left & right & hole).sum() / max(1, (left | right) [hole].sum()))
    return {"reobs_rgb": rgb.astype(np.float32), "reobs_mask": mask,
            "reobs_confidence": confidence, "observation_count": count,
            "stereo_agreement": stereo_agreement}
