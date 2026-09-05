#!/usr/bin/env python3
"""Prepare leak-free temporal ReObs prefills for residual diffusion inpainting.

Only neighboring Endoscope2 RGB/depth frames are projected into the calibrated
Endoscope1 coordinate system. Endoscope1 RGB is never opened or inspected.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stereocom.io_utils import (  # noqa: E402
    discover_e2_depth_frames,
    load_rgb,
    read_intrinsics,
    read_poses,
    resize_intrinsic,
)
from stereocom.warp import warp_e2_to_e1  # noqa: E402


E1_RGB_READ = False


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temporal-window", type=int, default=2)
    parser.add_argument("--min-observations", type=int, default=2)
    parser.add_argument("--projection-confidence-threshold", type=float, default=0.80)
    parser.add_argument("--depth-relative-threshold", type=float, default=0.03)
    parser.add_argument("--photometric-threshold", type=float, default=0.12)
    parser.add_argument("--temporal-tau", type=float, default=1.0)
    parser.add_argument(
        "--prefill-erode", type=int, default=1,
        help="erode reliable prefill mask by this many pixels before hard preservation",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visualizations", type=int, default=10)
    return parser.parse_args()


def validate_args(args):
    if args.temporal_window < 1:
        raise ValueError("--temporal-window must be >= 1")
    if args.min_observations < 1:
        raise ValueError("--min-observations must be >= 1")
    if not 0 <= args.projection_confidence_threshold <= 1:
        raise ValueError("--projection-confidence-threshold must be in [0, 1]")
    if args.depth_relative_threshold <= 0:
        raise ValueError("--depth-relative-threshold must be > 0")
    if args.photometric_threshold <= 0:
        raise ValueError("--photometric-threshold must be > 0")
    if args.temporal_tau <= 0:
        raise ValueError("--temporal-tau must be > 0")
    if args.prefill_erode < 0 or args.max_frames < 0 or args.visualizations < 0:
        raise ValueError("erosion, max frames, and visualization counts must be non-negative")


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_mask(path, target_hw):
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    if value.shape != target_hw:
        raise RuntimeError("Mask shape {} differs from target {}: {}".format(
            value.shape, target_hw, path))
    return value > 127


def load_rgb_uint8(path, target_hw):
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    if value.shape[:2] != target_hw:
        raise RuntimeError("RGB shape {} differs from target {}: {}".format(
            value.shape[:2], target_hw, path))
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb_uint8):
    value = np.asarray(rgb_uint8, dtype=np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(value, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Could not write {}".format(path))


def save_mask(path, mask):
    if not cv2.imwrite(str(path), np.asarray(mask, np.uint8) * 255):
        raise RuntimeError("Could not write {}".format(path))


def load_depth(path, target_hw):
    depth = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if depth.ndim != 2:
        raise RuntimeError("Depth must be HxW: {}".format(path))
    if depth.shape != target_hw:
        depth = cv2.resize(
            depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    valid = np.isfinite(depth) & (depth > 0)
    return depth, valid


def project_neighbor(rgb_path, depth_path, target_hw, k2, k1, poses, delta_t):
    rgb = load_rgb(rgb_path, target_hw)
    depth, depth_valid = load_depth(depth_path, target_hw)
    warped = warp_e2_to_e1(
        rgb, depth, depth_valid, k2, k1, poses[0], poses[1])
    warped_depth = np.asarray(warped["depth_mm"], dtype=np.float32)
    valid = (np.asarray(warped["valid_mask"], dtype=bool)
             & np.isfinite(warped_depth) & (warped_depth > 0))
    return {
        "rgb": np.asarray(warped["rgb"], dtype=np.float32),
        "depth": warped_depth,
        "valid": valid,
        "confidence": np.asarray(warped["confidence"], dtype=np.float32),
        "delta_t": int(delta_t),
    }


def select_reobservations(observations, missing_mask, args):
    height, width = missing_mask.shape
    if not observations:
        zeros = np.zeros((height, width), dtype=np.float32)
        return {
            "rgb": np.zeros((height, width, 3), dtype=np.float32),
            "candidate": np.zeros((height, width), dtype=bool),
            "confidence": zeros,
            "count": np.zeros((height, width), dtype=np.uint16),
            "delta_t": np.zeros((height, width), dtype=np.int16),
            "score": zeros,
        }

    rgbs = np.stack([item["rgb"] for item in observations])
    depths = np.stack([item["depth"] for item in observations])
    valids = np.stack([item["valid"] for item in observations])
    confidences = np.stack([item["confidence"] for item in observations])
    delta_ts = np.asarray([item["delta_t"] for item in observations], dtype=np.int16)

    finite_positive = np.isfinite(depths) & (depths > 0)
    valids &= finite_positive
    safe_depths = np.where(valids, depths, np.inf)
    front_depth = safe_depths.min(axis=0)
    finite_front = np.isfinite(front_depth) & (front_depth > 0)
    denominator = np.maximum(np.where(finite_front, front_depth, 1.0), 1e-6)
    relative_error = np.full_like(depths, np.inf, dtype=np.float32)
    np.divide(
        np.abs(depths - front_depth[None]), denominator[None], out=relative_error,
        where=valids & finite_front[None],
    )
    depth_consistent = valids & (relative_error < args.depth_relative_threshold)
    observation_count = depth_consistent.sum(axis=0).astype(np.uint16)

    masked_rgb = np.where(depth_consistent[..., None], rgbs, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        consensus_rgb = np.nanmedian(masked_rgb, axis=0)
    consensus_rgb = np.nan_to_num(consensus_rgb, nan=0.0, posinf=0.0, neginf=0.0)
    photometric_error = np.mean(np.abs(rgbs - consensus_rgb[None]), axis=-1)
    photometric_valid = depth_consistent & np.isfinite(photometric_error)
    photometric_valid &= photometric_error < args.photometric_threshold
    photometric_weight = np.clip(
        1.0 - photometric_error / args.photometric_threshold, 0.0, 1.0)
    temporal_weight = np.exp(
        -np.abs(delta_ts.astype(np.float32)) / args.temporal_tau)
    scores = confidences * temporal_weight[:, None, None] * photometric_weight
    scores = np.where(photometric_valid & np.isfinite(scores), scores, -np.inf)

    best_index = np.argmax(scores, axis=0)
    best_score = np.take_along_axis(scores, best_index[None], axis=0)[0]
    selected_valid = np.isfinite(best_score)
    gather_rgb = np.broadcast_to(best_index[None, ..., None], (1, height, width, 3))
    selected_rgb = np.take_along_axis(rgbs, gather_rgb, axis=0)[0]
    selected_confidence = np.take_along_axis(
        confidences, best_index[None], axis=0)[0]
    selected_delta_t = delta_ts[best_index]
    candidate = (
        missing_mask
        & (observation_count >= args.min_observations)
        & selected_valid
        & (selected_confidence >= args.projection_confidence_threshold)
    )
    selected_rgb[~selected_valid] = 0
    selected_confidence[~selected_valid] = 0
    selected_delta_t[~selected_valid] = 0
    best_score[~selected_valid] = 0
    return {
        "rgb": selected_rgb,
        "candidate": candidate,
        "confidence": selected_confidence.astype(np.float32),
        "count": observation_count,
        "delta_t": selected_delta_t.astype(np.int16),
        "score": best_score.astype(np.float32),
    }


def erode_mask(mask, radius):
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    size = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(np.asarray(mask, np.uint8), kernel, iterations=1) > 0


def save_visualization(path, panels):
    labelled = []
    for title, panel in panels:
        image = np.asarray(panel)
        if image.ndim == 2:
            image = np.repeat((image.astype(np.uint8) * 255)[..., None], 3, axis=2)
        else:
            image = image.astype(np.uint8)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 1, cv2.LINE_AA)
        labelled.append(bgr)
    cv2.imwrite(str(path), cv2.hconcat(labelled))


def required_output_files(row):
    keys = (
        "warped_rgb", "known_mask", "inpaint_mask", "reobs_rgb",
        "reobs_prefill_mask", "reobs_confidence", "reobs_observation_count",
        "reobs_selected_delta_t", "original_inpaint_mask",
    )
    return all(key in row and Path(row[key]).is_file() for key in keys)


def main():
    args = arguments()
    validate_args(args)
    input_root = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if input_root == output:
        raise ValueError("--output must differ from --input")
    source_manifest = json.loads((input_root / "warp_manifest.json").read_text())
    if not source_manifest.get("completed", True):
        raise RuntimeError("Input warp preparation is incomplete")
    if source_manifest.get("E1_RGB_READ") is not False:
        raise RuntimeError("Input manifest does not explicitly guarantee E1_RGB_READ=false")
    scene = Path(source_manifest["scene"]).resolve()
    target_hw = (int(source_manifest["height"]), int(source_manifest["width"]))

    all_pairs = discover_e2_depth_frames(scene)
    frame_to_position = {frame_id: pos for pos, (frame_id, _, _) in enumerate(all_pairs)}
    frame_to_paths = {frame_id: (rgb, depth) for frame_id, rgb, depth in all_pairs}
    source_frames = source_manifest["frames"][:args.max_frames or None]
    if not source_frames:
        raise RuntimeError("Input manifest contains no requested frames")
    unknown = [int(row["frame_id"]) for row in source_frames
               if int(row["frame_id"]) not in frame_to_position]
    if unknown:
        raise RuntimeError("Input frames missing from Endoscope2 RGB/depth: {}".format(unknown[:5]))

    first_rgb = cv2.imread(str(all_pairs[0][1]), cv2.IMREAD_GRAYSCALE)
    if first_rgb is None:
        raise FileNotFoundError(all_pairs[0][1])
    source_hw = first_rgb.shape
    intrinsics = read_intrinsics(scene / "K.txt")
    poses = read_poses(scene / "pose.txt")
    k1 = resize_intrinsic(intrinsics["K1_L"], source_hw, target_hw)
    k2 = resize_intrinsic(intrinsics["K2_L"], source_hw, target_hw)

    directory_names = (
        "warped_rgb", "known_mask", "inpaint_mask", "reobs_rgb",
        "reobs_prefill_mask", "reobs_confidence", "reobs_observation_count",
        "reobs_selected_delta_t", "original_inpaint_mask", "visualizations",
    )
    directories = {name: output / name for name in directory_names}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    settings = {
        "method": "ReObs-Guided Residual Inpainting",
        "scene": str(scene),
        "height": target_hw[0],
        "width": target_hw[1],
        "E1_RGB_READ": E1_RGB_READ,
        "temporal_window": int(args.temporal_window),
        "min_observations": int(args.min_observations),
        "projection_confidence_threshold": float(args.projection_confidence_threshold),
        "depth_relative_threshold": float(args.depth_relative_threshold),
        "photometric_threshold": float(args.photometric_threshold),
        "temporal_tau": float(args.temporal_tau),
        "prefill_erode": int(args.prefill_erode),
        "source_warp_manifest": str((input_root / "warp_manifest.json").resolve()),
    }
    manifest_path = output / "warp_manifest.json"
    previous = {}
    if manifest_path.is_file() and not args.overwrite:
        prior = json.loads(manifest_path.read_text())
        mismatches = [key for key, value in settings.items() if prior.get(key) != value]
        if mismatches:
            raise RuntimeError("Existing output uses different {}. Pass --overwrite or use "
                               "another output directory.".format(", ".join(mismatches)))
        previous = {int(row["frame_id"]): row for row in prior.get("frames", [])}

    records = []
    atomic_json(manifest_path, dict(settings, completed=False, frames=[]))
    for index, source_row in enumerate(source_frames, 1):
        frame_id = int(source_row["frame_id"])
        old = previous.get(frame_id)
        if old is not None and required_output_files(old) and not args.overwrite:
            records.append(old)
            atomic_json(manifest_path, dict(settings, completed=False, frames=records))
            print("[{}/{}] frame {:06d} already prepared".format(
                index, len(source_frames), frame_id), flush=True)
            continue

        original_rgb = load_rgb_uint8(source_row["warped_rgb"], target_hw)
        original_known = load_mask(source_row["known_mask"], target_hw)
        missing = load_mask(source_row["missing_mask"], target_hw)
        original_inpaint = load_mask(source_row["inpaint_mask"], target_hw)
        if np.any(missing & original_known):
            raise AssertionError("missing_mask overlaps known_mask for frame {}".format(frame_id))

        position = frame_to_position[frame_id]
        low = max(0, position - args.temporal_window)
        high = min(len(all_pairs), position + args.temporal_window + 1)
        observations = []
        neighbor_ids = []
        for neighbor_position in range(low, high):
            if neighbor_position == position:
                continue
            neighbor_id = all_pairs[neighbor_position][0]
            rgb_path, depth_path = frame_to_paths[neighbor_id]
            observations.append(project_neighbor(
                rgb_path, depth_path, target_hw, k2, k1, poses,
                neighbor_position - position,
            ))
            neighbor_ids.append(int(neighbor_id))

        selected = select_reobservations(observations, missing, args)
        stable_prefill = erode_mask(selected["candidate"], args.prefill_erode)
        assert not np.any(stable_prefill & original_known)
        assert np.all(stable_prefill <= missing)
        if stable_prefill.any():
            assert np.all(selected["confidence"][stable_prefill]
                          >= args.projection_confidence_threshold)
            assert np.all(selected["count"][stable_prefill] >= args.min_observations)

        reobs_uint8 = np.clip(selected["rgb"] * 255.0, 0, 255).astype(np.uint8)
        prefilled = original_rgb.copy()
        prefilled[stable_prefill] = reobs_uint8[stable_prefill]
        residual_inpaint = original_inpaint & ~stable_prefill
        residual_known = original_known | stable_prefill
        assert np.array_equal(prefilled[original_known], original_rgb[original_known])
        assert np.array_equal(residual_inpaint, original_inpaint & ~stable_prefill)
        assert np.array_equal(residual_known, original_known | stable_prefill)
        for array_name, array in (
            ("prefilled", prefilled), ("reobs", selected["rgb"]),
            ("confidence", selected["confidence"]), ("score", selected["score"]),
        ):
            if not np.isfinite(array).all():
                raise AssertionError("{} contains NaN/Inf for frame {}".format(
                    array_name, frame_id))

        filename = "frame_{:06d}.png".format(frame_id)
        array_filename = "frame_{:06d}.npy".format(frame_id)
        paths = {
            "warped_rgb": directories["warped_rgb"] / filename,
            "known_mask": directories["known_mask"] / filename,
            "inpaint_mask": directories["inpaint_mask"] / filename,
            "reobs_rgb": directories["reobs_rgb"] / filename,
            "reobs_prefill_mask": directories["reobs_prefill_mask"] / filename,
            "reobs_confidence": directories["reobs_confidence"] / array_filename,
            "reobs_observation_count": directories["reobs_observation_count"] / array_filename,
            "reobs_selected_delta_t": directories["reobs_selected_delta_t"] / array_filename,
            "original_inpaint_mask": directories["original_inpaint_mask"] / filename,
        }
        save_rgb(paths["warped_rgb"], prefilled)
        save_mask(paths["known_mask"], residual_known)
        save_mask(paths["inpaint_mask"], residual_inpaint)
        save_rgb(paths["reobs_rgb"], reobs_uint8)
        save_mask(paths["reobs_prefill_mask"], stable_prefill)
        save_mask(paths["original_inpaint_mask"], original_inpaint)
        np.save(paths["reobs_confidence"], selected["confidence"])
        np.save(paths["reobs_observation_count"], selected["count"])
        np.save(paths["reobs_selected_delta_t"], selected["delta_t"])

        original_pixels = int(original_inpaint.sum())
        missing_pixels = int(missing.sum())
        prefill_pixels = int(stable_prefill.sum())
        residual_pixels = int(residual_inpaint.sum())
        original_ratio = float(original_inpaint.mean())
        prefill_ratio = float(stable_prefill.mean())
        residual_ratio = float(residual_inpaint.mean())
        hole_reduction = (original_pixels - residual_pixels) / max(original_pixels, 1)
        prefill_over_missing = prefill_pixels / max(missing_pixels, 1)
        mean_confidence = (float(selected["confidence"][stable_prefill].mean())
                           if prefill_pixels else 0.0)
        mean_count = (float(selected["count"][stable_prefill].mean())
                      if prefill_pixels else 0.0)
        row = dict(source_row)
        row.update({key: str(path.resolve()) for key, path in paths.items()})
        row.update({
            "original_warped_rgb": str(Path(source_row["warped_rgb"]).resolve()),
            "original_known_mask": str(Path(source_row["known_mask"]).resolve()),
            "original_missing_mask": str(Path(source_row["missing_mask"]).resolve()),
            "original_inpaint_mask": str(paths["original_inpaint_mask"].resolve()),
            "neighbor_frame_ids": neighbor_ids,
            "original_inpaint_ratio": original_ratio,
            "geometric_missing_ratio": float(missing.mean()),
            "prefill_ratio": prefill_ratio,
            "prefill_over_missing_ratio": prefill_over_missing,
            "residual_inpaint_ratio": residual_ratio,
            "hole_reduction_ratio": float(hole_reduction),
            "mean_prefill_confidence": mean_confidence,
            "mean_prefill_observation_count": mean_count,
            "known_ratio": float(residual_known.mean()),
            "inpaint_ratio": residual_ratio,
        })
        records.append(row)
        atomic_json(manifest_path, dict(settings, completed=False, frames=records))

        if index <= args.visualizations:
            save_visualization(
                directories["visualizations"] / filename,
                (("original warp", original_rgb),
                 ("original inpaint", original_inpaint),
                 ("selected ReObs", reobs_uint8),
                 ("stable prefill", stable_prefill),
                 ("prefilled", prefilled),
                 ("residual inpaint", residual_inpaint)),
            )
        print(
            "frame {:06d}: original inpaint={:.3f} reobs prefill={:.3f} "
            "residual inpaint={:.3f} hole reduction={:.1%} mean confidence={:.3f} "
            "mean observations={:.2f}".format(
                frame_id, original_ratio, prefill_ratio, residual_ratio,
                hole_reduction, mean_confidence, mean_count), flush=True)

    reductions = np.asarray([row["hole_reduction_ratio"] for row in records])
    nonempty = [row for row in records if row["prefill_ratio"] > 0]
    summary = {
        "mean_original_inpaint_ratio": float(np.mean(
            [row["original_inpaint_ratio"] for row in records])),
        "mean_prefill_ratio": float(np.mean([row["prefill_ratio"] for row in records])),
        "mean_residual_inpaint_ratio": float(np.mean(
            [row["residual_inpaint_ratio"] for row in records])),
        "mean_hole_reduction": float(reductions.mean()),
        "median_hole_reduction": float(np.median(reductions)),
        "frames_with_zero_prefill": int(sum(row["prefill_ratio"] == 0 for row in records)),
        "mean_selected_confidence": (float(np.mean(
            [row["mean_prefill_confidence"] for row in nonempty])) if nonempty else 0.0),
        "mean_observation_count": (float(np.mean(
            [row["mean_prefill_observation_count"] for row in nonempty])) if nonempty else 0.0),
    }
    payload = dict(settings, completed=True, summary=summary, frames=records)
    atomic_json(manifest_path, payload)
    print("\nSCENE SUMMARY")
    for key, value in summary.items():
        print("{}: {}".format(key, value))
    print("Output manifest: {}".format(manifest_path))


if __name__ == "__main__":
    main()
