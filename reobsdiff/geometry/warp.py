"""Unified depth-aware warp; delegates splatting/z-buffering to GTCom."""

import numpy as np
from stereocom.warp import warp_e2_to_e1


def warp_to_pose(rgb, depth, K_src, T_src, K_dst, T_dst, valid_depth_mask=None,
                 depth_confidence=None, **kwargs):
    depth = np.asarray(depth, dtype=np.float32)
    K_src, K_dst = np.asarray(K_src), np.asarray(K_dst)
    valid = (np.isfinite(depth) & (depth > 0) if valid_depth_mask is None
             else np.asarray(valid_depth_mask, dtype=bool) & np.isfinite(depth) & (depth > 0))
    result = warp_e2_to_e1(rgb, depth, valid, np.asarray(K_src), np.asarray(K_dst),
                           np.asarray(T_src), np.asarray(T_dst), **kwargs)
    confidence = result["confidence"]
    if depth_confidence is not None:
        conf_warp = warp_e2_to_e1(
            np.repeat(np.asarray(depth_confidence, np.float32)[..., None], 3, axis=2),
            depth, valid, np.asarray(K_src), np.asarray(K_dst),
            np.asarray(T_src), np.asarray(T_dst), **kwargs)
        confidence = confidence * np.clip(conf_warp["rgb"][..., 0], 0, 1)
    h, w = depth.shape
    # Diagnostic source ownership map using nearest projected pixel and a
    # strict nearest-depth winner. RGB/depth/confidence still come from the
    # baseline's higher-quality bilinear splat above.
    yy, xx = np.mgrid[:h, :w]
    flat = np.flatnonzero(valid.reshape(-1))
    z = depth.reshape(-1)[flat]
    u = xx.reshape(-1)[flat].astype(np.float64)
    v = yy.reshape(-1)[flat].astype(np.float64)
    points = np.stack(((u - K_src[0, 2]) * z / K_src[0, 0],
                       (v - K_src[1, 2]) * z / K_src[1, 1], z, np.ones_like(z)))
    projected = np.linalg.inv(np.asarray(T_dst, np.float64)) @ np.asarray(T_src, np.float64) @ points
    with np.errstate(invalid="ignore", divide="ignore"):
        puf = K_dst[0, 0] * projected[0] / projected[2] + K_dst[0, 2]
        pvf = K_dst[1, 1] * projected[1] / projected[2] + K_dst[1, 2]
    finite = np.isfinite(projected).all(0) & np.isfinite(puf) & np.isfinite(pvf)
    pu = np.rint(np.where(finite, puf, -1)).astype(np.int64)
    pv = np.rint(np.where(finite, pvf, -1)).astype(np.int64)
    inside = finite & (projected[2] > 0) & (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
    destination = pv[inside] * w + pu[inside]
    order = np.argsort(projected[2, inside])[::-1]  # nearest assignment occurs last
    source_index_map = np.full(h * w, -1, dtype=np.int64)
    source_index_map[destination[order]] = flat[inside][order]
    return {
        "warped_rgb": result["rgb"], "warped_depth": result["depth_mm"],
        "valid_mask": result["valid_mask"], "confidence": confidence,
        "collision_confidence": confidence,
        "source_index_map": source_index_map.reshape(h, w),
        "projected_points": result["projected_points"],
    }
