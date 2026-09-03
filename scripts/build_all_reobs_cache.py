#!/usr/bin/env python3
"""Build ReObsDiff caches for every Task-2 scene with exact released poses.

This is the dataset-level entry point.  It delegates each scene to the tested
single-scene builder, preserves its atomic/resume behavior, validates every
completed manifest, and writes an aggregate manifest for training/auditing.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def read_completed_manifest(path):
    payload = json.loads(path.read_text())
    if not payload.get("completed"):
        raise RuntimeError("scene cache is incomplete: {}".format(path))
    if payload.get("target_view_rgb_reads") != 0:
        raise RuntimeError("nonzero target-view RGB reads: {}".format(path))
    if payload.get("target_view_depth_reads") != 0:
        raise RuntimeError("nonzero target-view depth reads: {}".format(path))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="Directory containing session_* Task-2 scenes")
    parser.add_argument("--output", required=True,
                        help="Root receiving one cache directory per scene")
    parser.add_argument("--config", default="configs/reobsdiff.yaml")
    parser.add_argument("--depth-source", choices=("gt", "foundationstereo"), default="gt")
    parser.add_argument("--visualizations", type=int, default=5)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    scenes = sorted(path for path in data_root.glob("session_*") if path.is_dir())
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    if not scenes:
        raise RuntimeError("no session_* scenes found below {}".format(data_root))
    output.mkdir(parents=True, exist_ok=True)

    fixed_overrides = (
        "depth_source={}".format(args.depth_source),
        "use_e1_camera_geometry=true",
        "pose_mode=e1_like",
        "translation_mm=0",
        "rotation_deg=0",
        "candidate_count=1",
        "min_overlap=0",
        "min_hole_ratio=0",
        "max_hole_ratio=1",
        "min_reobs_ratio=0",
        "target_hole_ratio=0.40",
    )
    aggregate_path = output / "dataset_manifest.json"
    aggregate = {
        "method": "reobsdiff", "completed": False,
        "data_root": str(data_root), "cache_root": str(output),
        "exact_dataset_pose": True, "all_frames_enabled": True,
        "depth_source": args.depth_source, "requested_scenes": len(scenes),
        "scenes": [], "failures": [],
    }
    atomic_json(aggregate_path, aggregate)

    for index, scene in enumerate(scenes, 1):
        scene_output = output / scene.name
        command = [
            sys.executable, str(repo / "scripts" / "build_reobs_cache.py"),
            "--scene", str(scene), "--output", str(scene_output),
            "--config", str(Path(args.config).resolve()), "--resume",
            "--visualizations", str(args.visualizations),
        ]
        if args.max_frames:
            command += ["--max-frames", str(args.max_frames)]
        for value in fixed_overrides:
            command += ["--set", value]
        print("\n=== [{}/{}] {} ===".format(index, len(scenes), scene.name), flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        try:
            subprocess.run(command, cwd=str(repo), check=True)
            manifest_path = scene_output / "reobs_manifest.json"
            payload = read_completed_manifest(manifest_path)
            row = {
                "scene": scene.name, "manifest": str(manifest_path.resolve()),
                "frames": len(payload.get("frames", [])),
                "skipped": int(payload.get("skipped", 0)),
                "skip_ratio": float(payload.get("skip_ratio", 0.0)),
            }
            if row["skipped"] != 0:
                raise RuntimeError("all-frames mode unexpectedly skipped frames in {}".format(scene.name))
            aggregate["scenes"].append(row)
        except Exception as error:
            aggregate["failures"].append({"scene": scene.name, "error": str(error)})
            atomic_json(aggregate_path, aggregate)
            if not args.continue_on_error:
                raise
        atomic_json(aggregate_path, aggregate)

    if args.dry_run:
        print("DRY RUN: discovered {} scenes".format(len(scenes)))
        return
    aggregate["completed"] = not aggregate["failures"] and len(aggregate["scenes"]) == len(scenes)
    aggregate["total_frames"] = sum(row["frames"] for row in aggregate["scenes"])
    aggregate["total_skipped"] = sum(row["skipped"] for row in aggregate["scenes"])
    atomic_json(aggregate_path, aggregate)
    if not aggregate["completed"]:
        raise RuntimeError("dataset cache incomplete; inspect {}".format(aggregate_path))
    print("\nALL_SCENES_CACHE_COMPLETE scenes={} frames={} skipped=0".format(
        len(aggregate["scenes"]), aggregate["total_frames"]), flush=True)
    print("aggregate manifest: {}".format(aggregate_path), flush=True)


if __name__ == "__main__":
    main()
