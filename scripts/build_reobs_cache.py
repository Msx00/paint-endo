#!/usr/bin/env python3
"""Build resumable, target-free reciprocal and re-observation training pairs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reobsdiff.config import load_config
from reobsdiff.depth.provider import make_depth_provider
from reobsdiff.geometry.reciprocal import build_geometry_mask, reciprocal_corruption
from reobsdiff.leakage import assert_no_target_view
from reobsdiff.pose.reobservable_sampler import ReobservablePoseSampler
from reobsdiff.reobs.target_builder import build_reobservation
from reobsdiff.visualization import save_cache_visualization
from stereocom.io_utils import discover_e2_pairs, load_rgb, read_intrinsics, read_poses, resize_intrinsic


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def boundary(mask):
    value = mask.astype(np.uint8)
    return cv2.dilate(value, np.ones((9, 9), np.uint8)) != cv2.erode(value, np.ones((9, 9), np.uint8))


def calibration_hash(scene):
    digest = hashlib.sha256()
    for name in ("K.txt", "pose.txt"):
        digest.update((scene / name).read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/reobsdiff.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--visualizations", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    scene, output = Path(args.scene).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = discover_e2_pairs(scene)
    if args.max_frames:
        pairs = pairs[:args.max_frames]
    header = cv2.imread(str(pairs[0][1]), cv2.IMREAD_GRAYSCALE)
    if header is None:
        raise FileNotFoundError(pairs[0][1])
    source_hw, target_hw = header.shape, (cfg.height, cfg.width)
    intrinsics, poses = read_intrinsics(scene / "K.txt"), read_poses(scene / "pose.txt")
    K_left = resize_intrinsic(intrinsics["K2_L"], source_hw, target_hw)
    T_left = poses[0]
    T_e1 = poses[1] if cfg.use_e1_camera_geometry else None
    provider = make_depth_provider(scene, cfg)
    sampler = ReobservablePoseSampler(cfg)
    images = {fid: (left, right) for fid, left, right in pairs}
    ordered_ids = [fid for fid, _, _ in pairs]
    settings = {"method": "reobsdiff", "scene": str(scene), "config": cfg.to_dict(),
                "depth_source": cfg.depth_source, "calibration_hash": calibration_hash(scene),
                "target_view_rgb_reads": 0, "target_view_depth_reads": 0}
    manifest_path = output / "reobs_manifest.json"
    old = {}
    if args.resume and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        for key, value in settings.items():
            if prior.get(key) != value:
                raise RuntimeError("resume manifest setting mismatch: {}".format(key))
        old = {int(row["frame_id"]): row for row in prior.get("frames", [])}
    records, skipped = [], 0
    atomic_json(manifest_path, dict(settings, completed=False, frames=[]))
    for position, frame_id in enumerate(ordered_ids):
        cached = old.get(frame_id)
        if cached and Path(cached["cache"]).is_file():
            records.append(cached)
            continue
        anchor_rgb = load_rgb(images[frame_id][0], target_hw).astype(np.float32) / 255.0
        anchor_depth = provider.get(frame_id, target_hw)
        observations, neighbor_ids = [], []
        if cfg.use_temporal_reobs:
            low, high = max(0, position - cfg.temporal_window), min(len(ordered_ids), position + cfg.temporal_window + 1)
            for neighbor_pos in range(low, high):
                if neighbor_pos == position:
                    continue
                neighbor_id = ordered_ids[neighbor_pos]
                depth = provider.get(neighbor_id, target_hw)
                observations.append({"rgb": load_rgb(images[neighbor_id][0], target_hw).astype(np.float32) / 255.0,
                    "depth": depth["depth"], "valid_depth_mask": depth["valid_depth_mask"],
                    "depth_confidence": depth["confidence"], "K": K_left, "T": T_left,
                    "delta_t": neighbor_pos - position, "view": "left", "view_weight": 1.0})
                neighbor_ids.append(neighbor_id)
        # Stereo is enabled only when released right depth and right extrinsics exist.
        stereo_used = False
        right_depth_path = scene / "endoscope2/depthR/frame_{:06d}.npy".format(frame_id)
        if cfg.use_stereo_reobs and right_depth_path.is_file() and 2 in poses and "K2_R" in intrinsics:
            right_depth = np.asarray(np.load(right_depth_path, allow_pickle=False), np.float32)
            right_depth = cv2.resize(right_depth, (cfg.width, cfg.height), interpolation=cv2.INTER_NEAREST)
            observations.append({"rgb": load_rgb(images[frame_id][1], target_hw).astype(np.float32) / 255.0,
                "depth": right_depth, "valid_depth_mask": np.isfinite(right_depth) & (right_depth > 0),
                "depth_confidence": np.ones(target_hw, np.float32),
                "K": resize_intrinsic(intrinsics["K2_R"], source_hw, target_hw), "T": poses[2],
                "delta_t": 0, "view": "right", "view_weight": 1.25})
            stereo_used = True
        selected = sampler.sample(anchor_rgb, anchor_depth["depth"], K_left, T_left,
                                  observations, T_e1, anchor_depth["valid_depth_mask"])
        if selected is None:
            skipped += 1
            diagnostic = sampler.last_diagnostics or {}
            atomic_json(manifest_path, dict(
                settings, completed=False, frames=records, skipped=skipped,
                processed=position + 1, skip_ratio=skipped / (position + 1)))
            print(("[{}/{}] frame {} skipped: best overlap={:.3f} hole={:.3f} "
                   "reobs={:.3f}; required overlap>={:.3f}, hole=[{:.3f},{:.3f}], "
                   "reobs>={:.3f}").format(
                       position + 1, len(pairs), frame_id,
                       diagnostic.get("overlap_ratio", float("nan")),
                       diagnostic.get("hole_ratio", float("nan")),
                       diagnostic.get("reobs_ratio", float("nan")),
                       cfg.min_overlap, cfg.min_hole_ratio, cfg.max_hole_ratio,
                       cfg.min_reobs_ratio), flush=True)
            continue
        virtual = selected["anchor_warp"]
        virtual_mask = build_geometry_mask(virtual["valid_mask"], cfg.mask_mode,
                                           cfg.mask_min_component, cfg.mask_dilation)
        reciprocal = reciprocal_corruption(anchor_rgb, anchor_depth["depth"], K_left, T_left,
            selected["T_virtual"], anchor_depth["valid_depth_mask"], cfg.mask_mode,
            cfg.mask_min_component, cfg.mask_dilation)
        reobs = build_reobservation(virtual["valid_mask"], observations, K_left,
                                    selected["T_virtual"], cfg)
        arrays = {"anchor_rgb": anchor_rgb, "anchor_depth": anchor_depth["depth"],
            "virtual_pose": selected["T_virtual"], "virtual_warp": virtual["warped_rgb"],
            "virtual_depth": virtual["warped_depth"], "virtual_valid": virtual["valid_mask"],
            "virtual_mask": virtual_mask, "reciprocal_rgb": reciprocal["corrupted_rgb"],
            "reciprocal_mask": reciprocal["reciprocal_mask"],
            "reciprocal_boundary": boundary(reciprocal["reciprocal_mask"]),
            "reobs_rgb": reobs["reobs_rgb"], "reobs_mask": reobs["reobs_mask"],
            "reobs_confidence": reobs["reobs_confidence"],
            "observation_count": reobs["observation_count"]}
        cache_path = output / "frames" / "frame_{:06d}.npz".format(frame_id)
        atomic_npz(cache_path, arrays)
        row = {"scene": scene.name, "frame_id": frame_id, "anchor_frame": frame_id,
            "cache": str(cache_path), "virtual_pose": selected["T_virtual"].tolist(),
            "overlap_ratio": selected["overlap_ratio"], "hole_ratio": selected["hole_ratio"],
            "reobs_ratio": selected["reobs_ratio"], "candidate_count": selected["candidate_count"],
            "neighbor_ids": neighbor_ids, "depth_source": cfg.depth_source,
            "calibration_hash": settings["calibration_hash"], "pose_mode": selected["pose_mode"],
            "rotation_delta_deg": selected["rotation_delta_deg"],
            "translation_delta_mm": selected["translation_delta_mm"],
            "stereo_requested": cfg.use_stereo_reobs, "stereo_used": stereo_used,
            "stereo_agreement": reobs["stereo_agreement"]}
        assert_no_target_view(row, "cache row")
        records.append(row)
        atomic_json(manifest_path, dict(settings, completed=False, frames=records,
                                        skipped=skipped, processed=position + 1,
                                        skip_ratio=skipped / (position + 1)))
        if len(records) <= args.visualizations:
            save_cache_visualization(output / "visualizations" / "frame_{:06d}".format(frame_id), arrays)
        print("[{}/{}] frame {} hole={:.3f} reobs={:.3f}".format(
            position + 1, len(pairs), frame_id, selected["hole_ratio"], selected["reobs_ratio"]), flush=True)
    payload = dict(settings, completed=True, frames=records, skipped=skipped,
                   skip_ratio=skipped / max(1, len(pairs)))
    assert_no_target_view(payload, "final cache manifest")
    atomic_json(manifest_path, payload)
    if not records:
        raise RuntimeError("all samples were skipped; relax pose constraints after inspecting geometry")
    print("CACHE COMPLETE frames={} skipped={} mean_reobs_coverage={:.6f}".format(
        len(records), skipped, np.mean([row["reobs_ratio"] for row in records])))


if __name__ == "__main__":
    main()
