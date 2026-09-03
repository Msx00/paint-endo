"""Geometry-only masks and forward/backward reciprocal corruption."""

import cv2
import numpy as np
from .warp import warp_to_pose


def build_geometry_mask(valid, mode="all_geometry", min_component=32, dilation=3):
    missing = ~np.asarray(valid, dtype=bool)
    if mode == "border":
        count, labels = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
        border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        missing = np.isin(labels, border[border != 0]) if count > 1 else missing
    elif mode != "all_geometry":
        raise ValueError("mask mode must be border or all_geometry")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), 8)
    keep = np.zeros_like(missing)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= int(min_component):
            keep |= labels == label
    if dilation > 1:
        k = int(dilation) | 1
        keep = cv2.dilate(keep.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return keep


def reciprocal_corruption(rgb, depth, K, T_anchor, T_virtual, valid_depth_mask=None,
                          mask_mode="all_geometry", min_component=32, dilation=3):
    forward = warp_to_pose(rgb, depth, K, T_anchor, K, T_virtual, valid_depth_mask)
    backward = warp_to_pose(
        forward["warped_rgb"], forward["warped_depth"], K, T_virtual, K, T_anchor,
        forward["valid_mask"])
    original_valid = (np.isfinite(depth) & (np.asarray(depth) > 0) if valid_depth_mask is None
                      else np.asarray(valid_depth_mask, bool))
    reciprocal_valid = backward["valid_mask"] & original_valid
    mask = build_geometry_mask(reciprocal_valid, mask_mode, min_component, dilation)
    corrupted = backward["warped_rgb"].copy()
    corrupted[mask] = 0
    return {"forward": forward, "backward": backward,
            "reciprocal_mask": mask, "corrupted_rgb": corrupted}
