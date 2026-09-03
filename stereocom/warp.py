"""Leak-free E2-to-E1 depth-aware reprojection."""

import numpy as np


def _depth_edge_confidence(depth, valid, sigma_mm):
    confidence = np.zeros_like(depth, dtype=np.float32)
    confidence[valid] = 1.0
    if sigma_mm <= 0:
        return confidence
    edge = np.zeros_like(depth, dtype=np.float32)
    dx = np.abs(depth[:, 1:] - depth[:, :-1])
    dy = np.abs(depth[1:, :] - depth[:-1, :])
    finite_x = valid[:, 1:] & valid[:, :-1] & np.isfinite(dx)
    finite_y = valid[1:, :] & valid[:-1, :] & np.isfinite(dy)
    dx = np.where(finite_x, dx, 0.0)
    dy = np.where(finite_y, dy, 0.0)
    edge[:, 1:] = np.maximum(edge[:, 1:], dx)
    edge[:, :-1] = np.maximum(edge[:, :-1], dx)
    edge[1:, :] = np.maximum(edge[1:, :], dy)
    edge[:-1, :] = np.maximum(edge[:-1, :], dy)
    confidence[valid] = np.exp(-edge[valid] / float(sigma_mm))
    return confidence


def warp_e2_to_e1(rgb, depth_mm, valid, k2, k1, c2w_e2, c2w_e1,
                  depth_tolerance_mm=1.0, min_splat_weight=0.05,
                  edge_sigma_mm=10.0, consistency_threshold_px=2.0):
    """Bilinearly splat E2 pixels into E1 with a nearest-surface buffer."""
    rgb = np.asarray(rgb, dtype=np.float32) / (255.0 if np.asarray(rgb).max() > 1 else 1.0)
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
    height, width = depth.shape
    if rgb.shape[:2] != (height, width):
        raise ValueError("RGB and depth grids differ")
    edge_confidence = _depth_edge_confidence(depth, valid, edge_sigma_mm)

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    flat_valid = valid.reshape(-1)
    z2 = depth.reshape(-1)[flat_valid]
    u2 = xx.reshape(-1)[flat_valid].astype(np.float32)
    v2 = yy.reshape(-1)[flat_valid].astype(np.float32)
    colors = rgb.reshape(-1, 3)[flat_valid]
    x2 = (u2 - k2[0, 2]) * z2 / k2[0, 0]
    y2 = (v2 - k2[1, 2]) * z2 / k2[1, 1]
    points2 = np.stack((x2, y2, z2, np.ones_like(z2)), axis=0)
    e1_from_e2 = np.linalg.inv(np.asarray(c2w_e1, dtype=np.float64)).dot(
        np.asarray(c2w_e2, dtype=np.float64)
    )
    points1 = e1_from_e2.dot(points2)
    z1 = points1[2]
    u1 = k1[0, 0] * points1[0] / z1 + k1[0, 2]
    v1 = k1[1, 1] * points1[1] / z1 + k1[1, 2]
    projected_valid = np.isfinite(points1).all(axis=0) & (z1 > 0)
    projected_valid &= (u1 > -1) & (u1 < width) & (v1 > -1) & (v1 < height)
    selected = np.flatnonzero(projected_valid)

    geometry_confidence = np.ones(selected.size, dtype=np.float32)
    if selected.size and consistency_threshold_px > 0:
        back = np.linalg.inv(e1_from_e2).dot(points1[:, selected])
        back_u = k2[0, 0] * back[0] / back[2] + k2[0, 2]
        back_v = k2[1, 1] * back[1] / back[2] + k2[1, 2]
        error = np.sqrt((back_u - u2[selected]) ** 2 + (back_v - v2[selected]) ** 2)
        keep = np.isfinite(error) & (error <= consistency_threshold_px)
        sigma = max(float(consistency_threshold_px), 1e-6)
        geometry_confidence = np.exp(-0.5 * (error[keep] / sigma) ** 2).astype(np.float32)
        selected = selected[keep]

    u = u1[selected].astype(np.float32)
    v = v1[selected].astype(np.float32)
    z = z1[selected].astype(np.float32)
    selected_colors = colors[selected].astype(np.float32)
    selected_edge = edge_confidence.reshape(-1)[flat_valid][selected]
    u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    pixel_count = height * width
    nearest = np.full(pixel_count, np.inf, dtype=np.float32)

    def neighbour(du, dv):
        uu, vv = u0 + du, v0 + dv
        weight = (1.0 - np.abs(u - uu)) * (1.0 - np.abs(v - vv))
        inside = (weight > 1e-8) & (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
        return uu, vv, weight.astype(np.float32), inside

    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        uu, vv, _, inside = neighbour(du, dv)
        indices = np.flatnonzero(inside)
        if indices.size:
            np.minimum.at(nearest, vv[indices] * width + uu[indices], z[indices])

    raw_sum = np.zeros(pixel_count, dtype=np.float32)
    weight_sum = np.zeros(pixel_count, dtype=np.float32)
    depth_sum = np.zeros(pixel_count, dtype=np.float32)
    color_sum = np.zeros((3, pixel_count), dtype=np.float32)
    tolerance = max(float(depth_tolerance_mm), 0.0)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        uu, vv, weight, inside = neighbour(du, dv)
        indices = np.flatnonzero(inside)
        if not indices.size:
            continue
        pixels = vv[indices] * width + uu[indices]
        visible = z[indices] <= nearest[pixels] + tolerance
        indices, pixels = indices[visible], pixels[visible]
        if not indices.size:
            continue
        delta = np.maximum(z[indices] - nearest[pixels], 0.0)
        front = np.exp(-delta / tolerance).astype(np.float32) if tolerance > 0 else (delta <= 1e-6)
        reliability = selected_edge[indices] * geometry_confidence[indices] * front
        raw_values, values = weight[indices], weight[indices] * reliability
        np.add.at(raw_sum, pixels, raw_values)
        np.add.at(weight_sum, pixels, values)
        np.add.at(depth_sum, pixels, values * z[indices])
        for channel in range(3):
            np.add.at(color_sum[channel], pixels, values * selected_colors[indices, channel])

    valid_pixels = (raw_sum >= min_splat_weight) & (weight_sum > 1e-8)
    confidence = np.zeros(pixel_count, dtype=np.float32)
    confidence[valid_pixels] = np.clip(raw_sum[valid_pixels], 0, 1) * np.clip(
        weight_sum[valid_pixels] / np.maximum(raw_sum[valid_pixels], 1e-8), 0, 1
    )
    output_rgb = np.zeros((pixel_count, 3), dtype=np.float32)
    output_depth = np.zeros(pixel_count, dtype=np.float32)
    output_rgb[valid_pixels] = (
        color_sum[:, valid_pixels] / weight_sum[valid_pixels][None]
    ).T
    output_depth[valid_pixels] = depth_sum[valid_pixels] / weight_sum[valid_pixels]
    return {
        "rgb": output_rgb.reshape(height, width, 3),
        "depth_mm": output_depth.reshape(height, width),
        "valid_mask": valid_pixels.reshape(height, width),
        "confidence": confidence.reshape(height, width),
        "projected_points": int(selected.size),
        "valid_pixels": int(valid_pixels.sum()),
    }
