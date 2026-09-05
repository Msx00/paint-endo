#!/usr/bin/env python3
"""Run the fixed LCM-step and DDIM sampling-quality ablation."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


EXPERIMENTS = (
    ("lcm", 4), ("lcm", 8), ("lcm", 12), ("ddim", 20), ("ddim", 30),
)
BASELINE_KEYS = (
    "prompt", "negative_prompt", "seed", "guidance_scale", "domain_lora_scale",
    "lcm_scale", "inpaint_exterior_only", "hard_composition", "model",
    "lcm_lora", "domain_lora",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--warp-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--domain-lora", required=True)
    parser.add_argument("--domain-lora-scale", type=float, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--lcm-lora", default=None)
    parser.add_argument("--lcm-scale", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--baseline-result-root", default="")
    parser.add_argument("--baseline-result-name", default="")
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--scene-glob", default="session_*")
    parser.add_argument("--evaluation-device", default="cpu")
    parser.add_argument("--evaluation-batch-size", type=int, default=4)
    parser.add_argument("--evaluation-workers", type=int, default=8)
    parser.add_argument(
        "--inpaint-exterior-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--hard-composition", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def checkpoint_result_prefix(path):
    path = Path(path)
    return "{}__{}".format(path.parent.name, path.name) if path.name.startswith(
        "checkpoint-") else path.name


def find_warp(warp_root, scene):
    candidates = (warp_root / scene / "warps", warp_root / scene)
    for candidate in candidates:
        manifest_path = candidate / "warp_manifest.json"
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text())
            method = str(payload.get("method", "")).lower()
            if "reobs" in method or "reobs_prefill" in candidate.name.lower():
                raise RuntimeError("ReObs-prefill warp is forbidden in sampler ablation: {}".format(
                    candidate))
            if payload.get("E1_RGB_READ") is not False:
                raise RuntimeError("Warp does not guarantee E1_RGB_READ=false: {}".format(candidate))
            return candidate.resolve()
    raise FileNotFoundError(
        "No existing original warp_manifest.json for {} below {}; refusing to run "
        "prepare_scene.py".format(scene, warp_root))


def load_consistent_baseline(root, result_name):
    paths = sorted(root.glob("*/results/{}/lcm/inference_manifest.json".format(result_name)))
    if not paths:
        raise FileNotFoundError("No baseline inference manifests for {} in {}".format(
            result_name, root))
    payloads = [json.loads(path.read_text()) for path in paths]
    reference = {key: payloads[0].get(key) for key in BASELINE_KEYS}
    for path, payload in zip(paths[1:], payloads[1:]):
        mismatches = [key for key, value in reference.items() if payload.get(key) != value]
        if mismatches:
            raise RuntimeError("Baseline settings differ at {}: {}".format(
                path, ", ".join(mismatches)))
    if not reference["hard_composition"]:
        raise RuntimeError("Baseline does not use hard composition")
    return reference, paths


def choose(explicit, inherited, fallback):
    return explicit if explicit is not None else inherited if inherited is not None else fallback


def run(command, cwd, env):
    print("\n$ {}".format(" ".join(str(part) for part in command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def inference_command(args, config, root, warp, output, sampler, steps, max_frames=0):
    script = root / ("infer_lcm.py" if sampler == "lcm" else "infer_ddim.py")
    command = [
        sys.executable, script, "--input", warp, "--output", output,
        "--model", config["model"], "--domain-lora", config["domain_lora"],
        "--domain-lora-scale", str(config["domain_lora_scale"]),
        "--steps", str(steps), "--guidance-scale", str(config["guidance_scale"]),
        "--batch-size", str(args.batch_size), "--seed", str(config["seed"]),
        "--device", args.device, "--prompt", config["prompt"],
        "--negative-prompt", config["negative_prompt"],
    ]
    if sampler == "lcm":
        command.extend(["--lcm-lora", config["lcm_lora"],
                        "--lcm-scale", str(config["lcm_scale"])])
    if config["inpaint_exterior_only"]:
        command.append("--inpaint-exterior-only")
    if not config["hard_composition"]:
        command.append("--disable-hard-composition")
    if max_frames:
        command.extend(["--max-frames", str(max_frames)])
    return command


def compare_reproduction(candidate_dir, baseline_dir, frame_ids):
    maximum, total, values, different = 0, 0, 0, 0
    for frame_id in frame_ids:
        name = "frame_{:06d}.png".format(frame_id)
        candidate = np.asarray(Image.open(candidate_dir / name).convert("RGB"), dtype=np.int16)
        baseline = np.asarray(Image.open(baseline_dir / name).convert("RGB"), dtype=np.int16)
        if candidate.shape != baseline.shape:
            raise RuntimeError("Reproduction shape mismatch for {}".format(name))
        difference = np.abs(candidate - baseline)
        maximum = max(maximum, int(difference.max()))
        total += int(difference.sum())
        values += difference.size
        different += int(np.count_nonzero(difference))
    result = {
        "max_absolute_uint8_difference": maximum,
        "mean_absolute_uint8_difference": total / max(values, 1),
        "different_values": different,
        "frames": len(frame_ids),
    }
    print("LCM4 BASELINE REPRODUCTION: {}".format(json.dumps(result)), flush=True)
    if maximum != 0 or different != 0:
        raise RuntimeError("LCM4 reproduction failed; stopping before full experiment")
    return result


def evaluate(args, root, output_root, result_name, sampler, env):
    report = output_root / "metrics" / result_name
    command = [
        sys.executable, root / "evaluate_metrics.py", "--outputs", output_root,
        "--data-root", args.data_root, "--report-dir", report,
        "--result-name", result_name, "--inference-subdir", sampler,
        "--prediction-kind", "final", "--device", args.evaluation_device,
        "--batch-size", str(args.evaluation_batch_size),
        "--workers", str(args.evaluation_workers),
    ]
    run(command, root, env)
    return json.loads((report / "summary.json").read_text())


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = arguments()
    root = Path(__file__).resolve().parents[1]
    data_root, warp_root = Path(args.data_root).resolve(), Path(args.warp_root).resolve()
    output_root = Path(args.output_root).resolve()
    baseline_root = Path(args.baseline_result_root).resolve() if args.baseline_result_root else warp_root
    domain_lora = Path(args.domain_lora).resolve()
    if not (domain_lora / "pytorch_lora_weights.safetensors").is_file():
        raise FileNotFoundError("Missing domain LoRA weights: {}".format(domain_lora))
    prefix = checkpoint_result_prefix(domain_lora)
    baseline_name = args.baseline_result_name or prefix
    inherited, baseline_paths = load_consistent_baseline(baseline_root, baseline_name)
    config = {
        "model": str(Path(choose(args.model, inherited["model"], root / "models/sd15-inpainting")).resolve()),
        "lcm_lora": str(Path(choose(args.lcm_lora, inherited["lcm_lora"], root / "models/lcm-lora-sdv1-5")).resolve()),
        "domain_lora": str(domain_lora),
        "domain_lora_scale": float(choose(args.domain_lora_scale, inherited["domain_lora_scale"], 0.8)),
        "lcm_scale": float(choose(args.lcm_scale, inherited["lcm_scale"], 1.0)),
        "guidance_scale": float(choose(args.guidance_scale, inherited["guidance_scale"], 1.5)),
        "seed": int(choose(args.seed, inherited["seed"], 6666)),
        "prompt": choose(args.prompt, inherited["prompt"], "surgical endoscopy image, realistic tissue"),
        "negative_prompt": choose(
            args.negative_prompt, inherited["negative_prompt"],
            "cartoon, illustration, text, watermark"),
        "inpaint_exterior_only": bool(choose(
            args.inpaint_exterior_only, inherited["inpaint_exterior_only"], True)),
        "hard_composition": bool(choose(args.hard_composition, inherited["hard_composition"], True)),
    }
    if Path(config["domain_lora"]) != Path(inherited["domain_lora"]):
        raise RuntimeError("Explicit --domain-lora differs from baseline manifest checkpoint")

    available = sorted(path.name for path in data_root.glob(args.scene_glob) if path.is_dir())
    scenes = args.scenes or available
    if not scenes:
        raise RuntimeError("No scenes selected")
    missing_data = [scene for scene in scenes if not (data_root / scene).is_dir()]
    if missing_data:
        raise FileNotFoundError("Unknown scenes: {}".format(missing_data))
    warps = {scene: find_warp(warp_root, scene) for scene in scenes}
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    settings = {
        "experiment_matrix": [{"sampler": s, "steps": n} for s, n in EXPERIMENTS],
        "checkpoint": str(domain_lora), "warp_root": str(warp_root),
        "baseline_result_root": str(baseline_root), "baseline_result_name": baseline_name,
        "scenes": scenes, "config": config, "baseline_manifests_checked": len(baseline_paths),
        "prepare_scene_called": False, "reobs_prefill_used": False,
    }
    (output_root / "sampler_ablation_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n")

    reproduction_scene = scenes[0]
    warp_payload = json.loads((warps[reproduction_scene] / "warp_manifest.json").read_text())
    reproduction_rows = warp_payload["frames"][:10]
    if len(reproduction_rows) < 10:
        raise RuntimeError("Reproduction scene has fewer than 10 warp frames")
    lcm4_name = "{}__lcm4".format(prefix)
    lcm4_output = output_root / reproduction_scene / "results" / lcm4_name / "lcm"
    candidate_paths = [lcm4_output / "final" / "frame_{:06d}.png".format(int(row["frame_id"]))
                       for row in reproduction_rows]
    if not all(path.is_file() for path in candidate_paths):
        run(inference_command(
            args, config, root, warps[reproduction_scene], lcm4_output,
            "lcm", 4, max_frames=10), root, env)
    baseline_dir = baseline_root / reproduction_scene / "results" / baseline_name / "lcm/final"
    reproduction = compare_reproduction(
        lcm4_output / "final", baseline_dir,
        [int(row["frame_id"]) for row in reproduction_rows])
    (output_root / "lcm4_reproduction.json").write_text(json.dumps(reproduction, indent=2) + "\n")

    summaries = {}
    for sampler, steps in EXPERIMENTS:
        result_name = "{}__{}{}".format(prefix, sampler, steps)
        for scene in scenes:
            inference_dir = output_root / scene / "results" / result_name / sampler
            run(inference_command(
                args, config, root, warps[scene], inference_dir,
                sampler, steps, max_frames=args.max_frames), root, env)
        summaries[(sampler, steps)] = evaluate(
            args, root, output_root, result_name, sampler, env)

    overall_rows, scene_rows = [], []
    baseline_summary = summaries[("lcm", 4)]
    baseline_overall = baseline_summary["overall"]
    for sampler, steps in EXPERIMENTS:
        result_name = "{}__{}{}".format(prefix, sampler, steps)
        summary = summaries[(sampler, steps)]
        timings = []
        for scene in scenes:
            manifest_path = output_root / scene / "results" / result_name / sampler / "inference_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            timings.extend(float(row["seconds_per_frame_batch"])
                           for row in manifest["frames"]
                           if row.get("seconds_per_frame_batch") is not None)
        overall = summary["overall"]
        overall_rows.append({
            "sampler": sampler, "steps": steps, "psnr": overall["psnr"],
            "ssim": overall["ssim"], "lpips": overall["lpips"],
            "delta_psnr": overall["psnr"] - baseline_overall["psnr"],
            "delta_ssim": overall["ssim"] - baseline_overall["ssim"],
            "delta_lpips": overall["lpips"] - baseline_overall["lpips"],
            "scene_count": overall["scenes"], "frame_count": overall["frames"],
            "sec_per_frame": float(np.mean(timings)) if timings else "",
        })
        for scene, values in summary["scenes"].items():
            scene_rows.append({
                "scene": scene, "sampler": sampler, "steps": steps,
                "psnr": values["psnr"], "ssim": values["ssim"],
                "lpips": values["lpips"], "frames": values["frames"],
            })
    summary_csv = output_root / "sampler_ablation_summary.csv"
    per_scene_csv = output_root / "sampler_ablation_per_scene.csv"
    write_csv(summary_csv, (
        "sampler", "steps", "psnr", "ssim", "lpips", "delta_psnr",
        "delta_ssim", "delta_lpips", "scene_count", "frame_count", "sec_per_frame"),
        overall_rows)
    write_csv(per_scene_csv, (
        "scene", "sampler", "steps", "psnr", "ssim", "lpips", "frames"), scene_rows)

    baseline_scenes = baseline_summary["scenes"]
    print("\nSAMPLER ABLATION SUMMARY")
    for row in overall_rows:
        print(row)
        if row["sampler"] == "lcm" and row["steps"] == 4:
            continue
        current = summaries[(row["sampler"], row["steps"])]["scenes"]
        common = sorted(set(baseline_scenes) & set(current))
        wins = {
            "psnr": sum(current[s]["psnr"] > baseline_scenes[s]["psnr"] for s in common),
            "ssim": sum(current[s]["ssim"] > baseline_scenes[s]["ssim"] for s in common),
            "lpips": sum(current[s]["lpips"] < baseline_scenes[s]["lpips"] for s in common),
        }
        print("scene wins vs LCM4: PSNR {psnr}/{n}, SSIM {ssim}/{n}, LPIPS {lpips}/{n}".format(
            n=len(common), **wins))
        for scene in common:
            if scene.startswith("session_004_scene_2"):
                print("  {}: {}".format(scene, current[scene]))
    print("Summary CSV: {}".format(summary_csv))
    print("Per-scene CSV: {}".format(per_scene_csv))


if __name__ == "__main__":
    main()
