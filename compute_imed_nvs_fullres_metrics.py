#!/usr/bin/env python3
"""Re-evaluate an iMED task-2 comparison run at native E1 resolution.

The evaluator reads ``manifest.json`` produced by ``test_imed_nvs_all.py``.
For every planned method/sequence task it:

1. locates the method-specific render directory;
2. pairs renders with the native ``endoscope1/L`` ground truth by sorted order;
3. bilinearly upsamples each render to the corresponding GT resolution;
4. evaluates inside ``non-tool E1 mask AND E2-to-E1 overlap mask``;
5. reports strict valid-pixel PSNR/SSIM and masked AlexNet LPIPS; and
6. defines success as a complete, readable, non-degenerate scene evaluation.

The main outputs are ``scene_metrics.csv`` and ``method_summary.csv``.  Scene
metrics are macro-averaged (each successfully evaluated sequence has equal
weight), matching the comparison coordinator's aggregation convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter


TASK2_ROOT = Path(__file__).resolve().parent
if str(TASK2_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK2_ROOT))

from imed_nvs_common import read_imed_intrinsics, read_imed_poses  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROTOCOL_VERSION = 1

# These adapters are the only layouts that differ from the standard
# ``model_path/test/ours_ITERATION/renders`` convention.
METHOD_ADAPTERS: Dict[str, str] = {
    "EndoGS": "endogs",
    "Free-SurGS": "free_surgs",
    "StructSplat": "structsplat",
}


@dataclass(frozen=True)
class PlannedTask:
    method: str
    sequence: str
    model_path: Path
    checkpoint_path: Path
    iteration: int


def image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no images found in: {directory}")
    return paths


def render_directory(task: PlannedTask) -> Path:
    adapter = METHOD_ADAPTERS.get(task.method, "standard")
    if adapter == "endogs":
        return task.checkpoint_path.parent / "render"
    if adapter == "free_surgs":
        return task.model_path / "test_results" / "renders"
    if adapter == "structsplat":
        raise ValueError(
            "StructSplat uses one joint ALL-sequences evaluation and does not expose "
            "a per-sequence render layout in the comparison manifest"
        )
    return task.model_path / "test" / f"ours_{task.iteration}" / "renders"


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def upsample_render_to_gt(render_path: Path, gt_size: Tuple[int, int]) -> torch.Tensor:
    """Load a render and bilinearly resize it to native GT (width, height)."""
    with Image.open(render_path) as image:
        image = image.convert("RGB")
        if image.size != gt_size:
            image = image.resize(gt_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_valid_tissue_mask(path: Optional[Path], size: Tuple[int, int]) -> torch.Tensor:
    if path is None:
        return torch.ones((1, size[1], size[0]), dtype=torch.float32)
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        # Dataset toolL is a foreground/tool mask: white means tool and is invalid.
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array < 128).float().unsqueeze(0)


def scale_intrinsics(
    intrinsics: np.ndarray,
    source_shape_hw: Tuple[int, int],
    target_shape_hw: Tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_shape_hw
    target_h, target_w = target_shape_hw
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError(
            f"invalid image shapes for intrinsics scaling: "
            f"{source_shape_hw} -> {target_shape_hw}"
        )
    scale = np.asarray(
        [
            [target_w / source_w, 0.0, 0.0],
            [0.0, target_h / source_h, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return scale @ intrinsics


def morphology(mask: np.ndarray, mode: str, size: int, iterations: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    filter_type = ImageFilter.MaxFilter if mode == "dilate" else ImageFilter.MinFilter
    for _ in range(iterations):
        image = image.filter(filter_type(size))
    return np.asarray(image, dtype=np.uint8) >= 128


def fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    flood = Image.fromarray(padded, mode="L")
    ImageDraw.floodfill(flood, (0, 0), 2)
    filled = np.asarray(flood, dtype=np.uint8) != 2
    return filled[1:-1, 1:-1]


def build_overlap_mask(sequence_root: Path, target_size: Tuple[int, int]) -> torch.Tensor:
    """Reproduce the attached E2-depth-to-E1 global overlap construction."""
    intrinsics = read_imed_intrinsics(sequence_root / "K.txt")
    poses = read_imed_poses(sequence_root / "pose.txt")
    k1 = intrinsics["K1_L"]
    k2 = intrinsics["K2_L"]
    c2w_cam2 = poses[0]
    c2w_cam1 = poses[1]
    w2c_cam1 = np.linalg.inv(c2w_cam1)

    depth_paths = sorted((sequence_root / "endoscope2" / "depthL").glob("*.npy"))
    if not depth_paths:
        raise FileNotFoundError(
            f"no E2 depth maps in {sequence_root / 'endoscope2' / 'depthL'}"
        )
    depth = np.squeeze(np.load(depth_paths[0]).astype(np.float32))
    if depth.ndim != 2:
        raise ValueError(f"source depth must be 2-D: {depth_paths[0]} -> {depth.shape}")
    depth_h, depth_w = depth.shape

    source_rgb_paths = image_files(sequence_root / "endoscope2" / "L")
    with Image.open(source_rgb_paths[0]) as source_rgb:
        source_rgb_shape = (source_rgb.height, source_rgb.width)
    k2_depth = scale_intrinsics(k2, source_rgb_shape, depth.shape)

    target_w, target_h = target_size
    target_rgb_paths = image_files(sequence_root / "endoscope1" / "L")
    with Image.open(target_rgb_paths[0]) as target_rgb:
        target_native_shape = (target_rgb.height, target_rgb.width)
    k1_target = scale_intrinsics(k1, target_native_shape, (target_h, target_w))

    u, v = np.meshgrid(
        np.arange(depth_w, dtype=np.float32),
        np.arange(depth_h, dtype=np.float32),
    )
    valid = np.isfinite(depth) & (depth > 0)
    x = (u - float(k2_depth[0, 2])) * depth / float(k2_depth[0, 0])
    y = (v - float(k2_depth[1, 2])) * depth / float(k2_depth[1, 1])
    points_cam2 = np.stack((x, y, depth), axis=-1)[valid]
    if points_cam2.size == 0:
        raise ValueError(f"no positive finite depth in {depth_paths[0]}")

    points_cam2_h = np.concatenate(
        (points_cam2, np.ones((points_cam2.shape[0], 1), dtype=np.float32)),
        axis=1,
    )
    points_world = (c2w_cam2 @ points_cam2_h.T).T
    points_cam1 = (w2c_cam1 @ points_world.T).T[:, :3]
    in_front = points_cam1[:, 2] > 1.0e-6
    points_cam1 = points_cam1[in_front]
    if points_cam1.size == 0:
        raise ValueError("no projected source points lie in front of E1")

    z1 = points_cam1[:, 2]
    u1 = np.rint(
        float(k1_target[0, 0]) * points_cam1[:, 0] / z1 + float(k1_target[0, 2])
    ).astype(np.int32)
    v1 = np.rint(
        float(k1_target[1, 1]) * points_cam1[:, 1] / z1 + float(k1_target[1, 2])
    ).astype(np.int32)
    inside = (u1 >= 0) & (u1 < target_w) & (v1 >= 0) & (v1 < target_h)
    if not np.any(inside):
        raise ValueError("no E2 source points project inside the E1 image")

    overlap = np.zeros((target_h, target_w), dtype=np.bool_)
    overlap[v1[inside], u1[inside]] = True
    overlap = morphology(overlap, "dilate", size=3, iterations=2)
    # scipy.ndimage.binary_closing(..., 11x11, iterations=2)
    overlap = morphology(overlap, "dilate", size=11, iterations=2)
    overlap = morphology(overlap, "erode", size=11, iterations=2)
    overlap = fill_binary_holes(overlap)
    if not np.any(overlap):
        raise ValueError(f"empty E2-to-E1 overlap mask for {sequence_root.name}")
    return torch.from_numpy(overlap.copy()).float().unsqueeze(0)


def gaussian_window(
    channels: int,
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates -= window_size // 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).unsqueeze(0).unsqueeze(0)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> float:
    mask3 = mask.expand_as(prediction)
    denominator = mask3.sum().clamp_min(1.0)
    mse = ((prediction - target).square() * mask3).sum() / denominator
    return float((-10.0 * torch.log10(mse + eps)).item())


def masked_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
    eps: float = 1.0e-8,
) -> float:
    """SSIM map averaged only over valid pixels, as in the supplied code."""
    prediction = prediction.unsqueeze(0)
    target = target.unsqueeze(0)
    mask = mask.unsqueeze(0)
    channels = prediction.shape[1]
    window = gaussian_window(
        channels, window_size, 1.5, prediction.device, prediction.dtype
    )
    padding = window_size // 2

    mu1 = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu2 = F.conv2d(target, window, padding=padding, groups=channels)
    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu12 = mu1 * mu2
    sigma1_sq = (
        F.conv2d(prediction.square(), window, padding=padding, groups=channels)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(target.square(), window, padding=padding, groups=channels)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(prediction * target, window, padding=padding, groups=channels)
        - mu12
    )
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = (
        (2.0 * mu12 + c1)
        * (2.0 * sigma12 + c2)
        / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + eps)
    )
    mask3 = mask.expand_as(ssim_map)
    return float((ssim_map * mask3).sum().div(mask3.sum().clamp_min(1.0)).item())


class LPIPSEvaluator:
    def __init__(self, device: torch.device, disabled: bool = False) -> None:
        self.device = device
        self.backend = "disabled" if disabled else "unavailable"
        self.model = None
        if disabled:
            return
        try:
            import lpips  # type: ignore

            self.model = lpips.LPIPS(net="alex").eval().to(device)
            self.backend = "lpips-alex"
        except Exception as error:
            raise RuntimeError(
                "LPIPS could not be initialized. Run this script in the SurgicalGS "
                "environment (or install the lpips package), or use --no-lpips only "
                f"for a PSNR/SSIM debug run. Original error: {error}"
            ) from error

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Optional[float]:
        if self.model is None:
            return None
        valid = mask.expand_as(prediction)
        prediction = (prediction * valid).unsqueeze(0)
        target = (target * valid).unsqueeze(0)
        with torch.no_grad():
            # Match the supplied lpips_score default: inputs are passed as [0,1]
            # without LPIPS' optional normalize=True conversion.
            value = self.model(prediction, target, normalize=False)
        return float(torch.as_tensor(value).mean().item())


def finite_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


def mask_paths_for_gt(sequence_root: Path, gt_paths: Sequence[Path]) -> List[Optional[Path]]:
    mask_dir = sequence_root / "endoscope1" / "toolL"
    if not mask_dir.is_dir():
        return [None] * len(gt_paths)
    available = {path.name: path for path in image_files(mask_dir)} if any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in mask_dir.iterdir()
    ) else {}
    if not available:
        return [None] * len(gt_paths)
    missing = [path.name for path in gt_paths if path.name not in available]
    if missing:
        raise ValueError(
            f"tool-mask set is incomplete ({len(missing)} missing), e.g. {missing[:3]}"
        )
    return [available[path.name] for path in gt_paths]


def is_degenerate(image: torch.Tensor) -> bool:
    dynamic_range = float((image.max() - image.min()).item())
    intensity_std = float(image.std().item())
    return dynamic_range <= (1.0 / 255.0) or intensity_std <= (0.5 / 255.0)


def evaluate_task(
    task: PlannedTask,
    data_root: Path,
    device: torch.device,
    lpips_evaluator: LPIPSEvaluator,
    overlap_cache: Dict[Tuple[str, int, int], torch.Tensor],
) -> Dict[str, object]:
    if task.sequence == "ALL":
        raise ValueError("joint ALL-sequences tasks cannot be paired to one E1 GT directory")
    sequence_root = data_root / task.sequence
    gt_paths = image_files(sequence_root / "endoscope1" / "L")
    render_dir = render_directory(task)
    render_paths = image_files(render_dir)
    if len(render_paths) != len(gt_paths):
        raise ValueError(
            f"incomplete render set: {len(render_paths)} renders vs "
            f"{len(gt_paths)} native E1 GT frames"
        )
    tool_mask_paths = mask_paths_for_gt(sequence_root, gt_paths)

    with Image.open(gt_paths[0]) as first_gt:
        target_size = first_gt.size
    cache_key = (task.sequence, target_size[0], target_size[1])
    if cache_key not in overlap_cache:
        overlap_cache[cache_key] = build_overlap_mask(sequence_root, target_size)
    overlap_mask_cpu = overlap_cache[cache_key]

    per_view: List[Dict[str, object]] = []
    degenerate_views = 0
    missing_tool_masks = sum(path is None for path in tool_mask_paths)
    for index, (render_path, gt_path, tool_mask_path) in enumerate(
        zip(render_paths, gt_paths, tool_mask_paths)
    ):
        with Image.open(gt_path) as gt_image:
            gt_size = gt_image.size
        if gt_size != target_size:
            raise ValueError(
                f"inconsistent native GT size: {gt_path} is {gt_size}, expected {target_size}"
            )
        with Image.open(render_path) as render_image:
            render_size = render_image.size

        prediction = upsample_render_to_gt(render_path, gt_size)
        target = load_rgb(gt_path)
        degenerate = is_degenerate(prediction)
        degenerate_views += int(degenerate)
        tissue_mask = load_valid_tissue_mask(tool_mask_path, gt_size)
        mask = tissue_mask * overlap_mask_cpu
        valid_pixels = int(mask.sum().item())
        if valid_pixels == 0:
            raise ValueError(f"empty joint mask for {gt_path.name}")

        prediction = prediction.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.no_grad():
            psnr_value = masked_psnr(prediction, target, mask)
            ssim_value = masked_ssim(prediction, target, mask)
            lpips_value = lpips_evaluator(prediction, target, mask)
        per_view.append(
            {
                "index": index,
                "render": str(render_path),
                "gt": str(gt_path),
                "tool_mask": str(tool_mask_path) if tool_mask_path else None,
                "render_width": render_size[0],
                "render_height": render_size[1],
                "gt_width": gt_size[0],
                "gt_height": gt_size[1],
                "valid_pixels": valid_pixels,
                "valid_ratio": valid_pixels / float(gt_size[0] * gt_size[1]),
                "psnr": psnr_value,
                "ssim": ssim_value,
                "lpips": lpips_value,
                "degenerate": degenerate,
            }
        )

    if degenerate_views == len(per_view):
        raise ValueError(f"all {degenerate_views} rendered frames are degenerate")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "success",
        "method": task.method,
        "sequence": task.sequence,
        "render_dir": str(render_dir),
        "gt_dir": str(sequence_root / "endoscope1" / "L"),
        "tool_mask_dir": str(sequence_root / "endoscope1" / "toolL"),
        "resize": "render bilinear-upsampled to each native GT resolution",
        "mask_protocol": "E1 non-tool mask AND calibrated E2-to-E1 overlap mask",
        "aggregation": "arithmetic mean over frames",
        "lpips_backend": lpips_evaluator.backend,
        "num_views": len(per_view),
        "degenerate_views": degenerate_views,
        "missing_tool_masks": missing_tool_masks,
        "metrics": {
            "psnr": finite_mean(row["psnr"] for row in per_view),
            "ssim": finite_mean(row["ssim"] for row in per_view),
            "lpips": finite_mean(row["lpips"] for row in per_view),
        },
        "per_view": per_view,
    }


def load_manifest(run_dir: Path) -> Mapping[str, object]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing comparison manifest: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("planned_tasks"), list):
        raise ValueError(f"invalid comparison manifest: {path}")
    return value


def planned_tasks(manifest: Mapping[str, object]) -> List[PlannedTask]:
    tasks: List[PlannedTask] = []
    for raw in manifest["planned_tasks"]:  # type: ignore[index]
        tasks.append(
            PlannedTask(
                method=str(raw["method"]),
                sequence=str(raw["sequence"]),
                model_path=Path(raw["model_path"]),
                checkpoint_path=Path(raw["checkpoint_path"]),
                iteration=int(raw["iteration"]),
            )
        )
    return tasks


def expected_task_rows(
    run_dir: Path,
    manifest: Mapping[str, object],
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Return the original run's denominator, including skipped checkpoints."""
    summary_path = run_dir / "summary.csv"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return {
            (str(row["method"]), str(row["sequence"])): dict(row)
            for row in rows
            if row.get("method") and row.get("method") != "__coordinator__"
        }

    # Fallback for an older run directory without summary.csv.
    methods = [str(value) for value in manifest.get("methods", [])]
    sequences = [str(value) for value in manifest.get("sequences", [])]
    expected: Dict[Tuple[str, str], Dict[str, str]] = {}
    for method in methods:
        method_sequences = ["ALL"] if method == "StructSplat" else sequences
        for sequence in method_sequences:
            expected[(method, sequence)] = {
                "method": method,
                "sequence": sequence,
                "status": "expected",
                "message": "",
            }
    return expected


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    all_results: Sequence[Mapping[str, object]],
    method_order: Sequence[str],
) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for method in method_order:
        expected = [row for row in all_results if row["method"] == method]
        successful = [row for row in expected if row["status"] == "success"]

        def metric_mean(name: str) -> Optional[float]:
            return finite_mean(
                row.get("metrics", {}).get(name)  # type: ignore[union-attr]
                for row in successful
            )

        summaries.append(
            {
                "method": method,
                # Requested paper metric order: PSNR, SSIM, LPIPS, success rate.
                "psnr": metric_mean("psnr"),
                "ssim": metric_mean("ssim"),
                "lpips": metric_mean("lpips"),
                "success_rate": len(successful) / len(expected) if expected else 0.0,
                "successful_scenes": len(successful),
                "expected_scenes": len(expected),
            }
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=TASK2_ROOT / "test_runs" / "qualified_methods",
        help="Completed test_imed_nvs_all.py run containing manifest.json.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Override manifest data_root (normally unnecessary).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Default: RUN_DIR/fullres_metrics.",
    )
    parser.add_argument("--methods", nargs="+", help="Optional subset of method names.")
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Optional subset of sequence names (useful for a smoke test).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific device such as cuda:1.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse successful task JSON files.")
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Debug only; the requested paper evaluation requires LPIPS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = load_manifest(run_dir)
    data_root = (
        args.data_root.resolve()
        if args.data_root is not None
        else Path(str(manifest["data_root"])).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "fullres_metrics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    tasks = planned_tasks(manifest)
    manifest_methods = [str(value) for value in manifest.get("methods", [])]
    expected = expected_task_rows(run_dir, manifest)
    if args.methods:
        requested = set(args.methods)
        unknown = requested.difference(manifest_methods)
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
        tasks = [task for task in tasks if task.method in requested]
        expected = {
            key: value for key, value in expected.items() if key[0] in requested
        }
        method_order = [method for method in manifest_methods if method in requested]
    else:
        method_order = manifest_methods or list(dict.fromkeys(task.method for task in tasks))
    if args.sequences:
        requested_sequences = set(args.sequences)
        known_sequences = {str(value) for value in manifest.get("sequences", [])}
        known_sequences.add("ALL")
        unknown_sequences = requested_sequences.difference(known_sequences)
        if unknown_sequences:
            raise ValueError(f"unknown sequences: {sorted(unknown_sequences)}")
        tasks = [task for task in tasks if task.sequence in requested_sequences]
        expected = {
            key: value
            for key, value in expected.items()
            if key[1] in requested_sequences
        }

    print(f"Run directory : {run_dir}")
    print(f"Dataset       : {data_root}")
    print(f"Output        : {output_dir}")
    print(f"Device        : {device}")
    print(f"Runnable tasks: {len(tasks)}")
    print(f"Expected tasks: {len(expected)} (success-rate denominator)")
    print("Protocol      : render -> bilinear native E1 size; non-tool & overlap mask")
    lpips_evaluator = LPIPSEvaluator(device, disabled=args.no_lpips)
    print(f"LPIPS backend : {lpips_evaluator.backend}")

    overlap_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}
    results: List[Dict[str, object]] = []
    task_output_root = output_dir / "tasks"
    for task_index, task in enumerate(tasks, start=1):
        task_path = task_output_root / safe_name(task.method) / f"{safe_name(task.sequence)}.json"
        if args.resume and task_path.is_file():
            with task_path.open("r", encoding="utf-8") as handle:
                previous = json.load(handle)
            if (
                previous.get("protocol_version") == PROTOCOL_VERSION
                and previous.get("status") == "success"
            ):
                results.append(previous)
                print(
                    f"[{task_index:03d}/{len(tasks):03d}] RESUME "
                    f"{task.method} | {task.sequence}"
                )
                continue

        print(f"[{task_index:03d}/{len(tasks):03d}] START  {task.method} | {task.sequence}")
        try:
            result = evaluate_task(
                task, data_root, device, lpips_evaluator, overlap_cache
            )
            metrics = result["metrics"]
            print(
                f"                         OK     PSNR={metrics['psnr']:.6f} "
                f"SSIM={metrics['ssim']:.6f} LPIPS={metrics['lpips']}"
            )
        except Exception as error:
            result = {
                "protocol_version": PROTOCOL_VERSION,
                "status": "failed",
                "method": task.method,
                "sequence": task.sequence,
                "metrics": {"psnr": None, "ssim": None, "lpips": None},
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"                         FAILED {result['error']}")
        atomic_json(task_path, result)
        results.append(result)

        # Keep resumable aggregate state after every task.
        atomic_json(output_dir / "all_results.json", results)

    # The coordinator manifest contains runnable tasks only.  Preserve skipped
    # or checkpoint-missing tasks from its summary so success rates use the
    # actual requested denominator rather than only the launchable subset.
    evaluated_keys = {(str(row["method"]), str(row["sequence"])) for row in results}
    for key, original in expected.items():
        if key in evaluated_keys:
            continue
        method, sequence = key
        original_status = original.get("status", "not-runnable")
        original_message = original.get("message", "")
        results.append(
            {
                "protocol_version": PROTOCOL_VERSION,
                "status": "failed",
                "method": method,
                "sequence": sequence,
                "metrics": {"psnr": None, "ssim": None, "lpips": None},
                "error": (
                    f"not runnable in original comparison run "
                    f"(status={original_status}): {original_message}"
                ).rstrip(),
            }
        )
    results.sort(key=lambda row: (str(row["method"]), str(row["sequence"])))
    atomic_json(output_dir / "all_results.json", results)

    scene_rows: List[Dict[str, object]] = []
    for row in results:
        metrics = row.get("metrics", {})
        scene_rows.append(
            {
                "method": row["method"],
                "sequence": row["sequence"],
                "status": row["status"],
                "psnr": metrics.get("psnr"),  # type: ignore[union-attr]
                "ssim": metrics.get("ssim"),  # type: ignore[union-attr]
                "lpips": metrics.get("lpips"),  # type: ignore[union-attr]
                "num_views": row.get("num_views"),
                "degenerate_views": row.get("degenerate_views"),
                "missing_tool_masks": row.get("missing_tool_masks"),
                "error": row.get("error", ""),
            }
        )
    write_csv(
        output_dir / "scene_metrics.csv",
        (
            "method", "sequence", "status", "psnr", "ssim", "lpips",
            "num_views", "degenerate_views", "missing_tool_masks", "error",
        ),
        scene_rows,
    )

    summaries = summarize(results, method_order)
    write_csv(
        output_dir / "method_summary.csv",
        (
            "method", "psnr", "ssim", "lpips", "success_rate",
            "successful_scenes", "expected_scenes",
        ),
        summaries,
    )
    atomic_json(
        output_dir / "summary.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "run_dir": str(run_dir),
            "data_root": str(data_root),
            "device": str(device),
            "resize": "bilinear render upsampling to native endoscope1 GT size",
            "mask": "E1 non-tool AND calibrated E2-to-E1 overlap",
            "scene_aggregation": "macro average across successful scenes",
            "success": "complete, readable, non-fully-degenerate scene evaluation",
            "methods": summaries,
        },
    )

    print("\nMethod summary (PSNR, SSIM, LPIPS, success rate):")
    for row in summaries:
        psnr = "--" if row["psnr"] is None else f"{row['psnr']:.6f}"
        ssim = "--" if row["ssim"] is None else f"{row['ssim']:.6f}"
        lpips = "--" if row["lpips"] is None else f"{row['lpips']:.6f}"
        print(
            f"{row['method']:<26} PSNR={psnr} SSIM={ssim} LPIPS={lpips} "
            f"Success={100.0 * row['success_rate']:.1f}% "
            f"({row['successful_scenes']}/{row['expected_scenes']})"
        )
    print(f"\nSaved: {output_dir / 'method_summary.csv'}")


if __name__ == "__main__":
    main()
