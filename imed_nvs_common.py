"""Shared, read-only loader utilities for the iMED task-2 NVS dataset.

The individual comparison repositories use slightly different ``CameraInfo``
named tuples.  This module keeps parsing and split semantics in one place and
only passes fields supported by the caller's ``CameraInfo`` type.

Dataset convention used by all adapters (official iMED NVS protocol):

* endoscope2/L is the source/training view;
* endoscope1/L is the held-out target/test view;
* frame ids define time, so equal frame names from both endoscopes have the
  same normalized timestamp;
* pose.txt stores ``camera_id tx ty tz qx qy qz qw`` camera-to-world poses,
  where camera id 0 is endoscope2/L and camera id 1 is endoscope1/L;
* toolL is a foreground/tool mask and is inverted to obtain valid tissue;
* depthL contains metric ``.npy`` depth maps.

No files are created in the dataset directory by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as torch_f
from PIL import Image


@dataclass(frozen=True)
class IMedFrame:
    uid: int
    endoscope: str
    image_path: Path
    depth_path: Optional[Path]
    tool_mask_path: Optional[Path]
    intrinsic: np.ndarray
    c2w: np.ndarray
    time: float


class IMedCameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: torch.Tensor
    image_path: str
    image_name: str
    width: int
    height: int
    time: float
    depth: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    Zfar: Optional[float] = None
    Znear: Optional[float] = None
    pc: Optional[torch.Tensor] = None
    intrinsic: Optional[np.ndarray] = None


def is_imed_nvs_sequence(path: Union[str, Path]) -> bool:
    root = Path(path)
    return all(
        candidate.exists()
        for candidate in (
            root / "K.txt",
            root / "pose.txt",
            root / "points3d.ply",
            root / "endoscope1" / "L",
            root / "endoscope2" / "L",
        )
    )


def _quaternion_xyzw_to_matrix(values: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("pose.txt contains a zero-length quaternion")
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def read_imed_intrinsics(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Parse the four labelled 3x3 matrices in K.txt."""
    matrices: Dict[str, np.ndarray] = {}
    label: Optional[str] = None
    rows: List[List[float]] = []

    def commit() -> None:
        nonlocal rows
        if label is not None and rows:
            matrix = np.asarray(rows, dtype=np.float32)
            if matrix.shape != (3, 3):
                raise ValueError(f"{label} in {path} is not a 3x3 matrix")
            matrices[label] = matrix
        rows = []

    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                commit()
                words = line[1:].strip().split()
                label = words[0] if words and words[0].startswith("K") else None
            elif label is not None:
                rows.append([float(item) for item in line.split()])
    commit()

    required = {"K1_L", "K2_L"}
    missing = required.difference(matrices)
    if missing:
        raise KeyError(f"Missing {sorted(missing)} in {path}")
    return matrices


def read_imed_poses(path: Union[str, Path]) -> Dict[int, np.ndarray]:
    poses: Dict[int, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(item) for item in line.split()]
            if len(values) != 8:
                raise ValueError(f"Invalid pose at {path}:{line_number}; expected 8 values")
            camera_id = int(values[0])
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = _quaternion_xyzw_to_matrix(values[4:8])
            c2w[:3, 3] = np.asarray(values[1:4], dtype=np.float32)
            poses[camera_id] = c2w
    if 0 not in poses or 1 not in poses:
        raise KeyError(f"{path} must contain camera ids 0 and 1")
    return poses


def _frame_number(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    if not digits:
        raise ValueError(f"Cannot extract frame number from {path.name}")
    return int(digits)


def _discover_frames(root: Path) -> Tuple[List[IMedFrame], List[IMedFrame]]:
    intrinsics = read_imed_intrinsics(root / "K.txt")
    poses = read_imed_poses(root / "pose.txt")
    per_scope: Dict[str, List[Path]] = {}
    all_frame_numbers: Set[int] = set()
    for endoscope in ("endoscope2", "endoscope1"):
        images = sorted((root / endoscope / "L").glob("frame_*.png"), key=_frame_number)
        if not images:
            raise FileNotFoundError(f"No frame_*.png images in {root / endoscope / 'L'}")
        per_scope[endoscope] = images
        all_frame_numbers.update(_frame_number(path) for path in images)

    ordered_numbers = sorted(all_frame_numbers)
    time_by_number = {
        frame_number: (index / max(1, len(ordered_numbers) - 1))
        for index, frame_number in enumerate(ordered_numbers)
    }

    splits: List[List[IMedFrame]] = [[], []]
    next_uid = 0
    # The released Task-2 NVS data uses pose id 0 for the training/source
    # endoscope2 camera and pose id 1 for the held-out endoscope1 camera.
    # This mapping is intentionally not inferred from the numeric suffix of
    # the directory name.
    split_specs = (
        ("endoscope2", "K2_L", 0),
        ("endoscope1", "K1_L", 1),
    )
    for scope_index, (endoscope, intrinsic_key, pose_id) in enumerate(split_specs):
        intrinsic = intrinsics[intrinsic_key]
        c2w = poses[pose_id]
        for image_path in per_scope[endoscope]:
            depth_candidate = root / endoscope / "depthL" / f"{image_path.stem}.npy"
            mask_candidate = root / endoscope / "toolL" / image_path.name
            splits[scope_index].append(
                IMedFrame(
                    uid=next_uid,
                    endoscope=endoscope,
                    image_path=image_path,
                    depth_path=depth_candidate if depth_candidate.is_file() else None,
                    tool_mask_path=mask_candidate if mask_candidate.is_file() else None,
                    intrinsic=intrinsic.copy(),
                    c2w=c2w.copy(),
                    time=float(time_by_number[_frame_number(image_path)]),
                )
            )
            next_uid += 1
    return splits[0], splits[1]


def _resize_tensor(tensor: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    if tuple(tensor.shape[-2:]) == size:
        return tensor
    source = tensor.unsqueeze(0)
    kwargs = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return torch_f.interpolate(source, **kwargs).squeeze(0)


def _load_frame_tensors(
    frame: IMedFrame,
    resolution_scale: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, np.ndarray]:
    if resolution_scale <= 0:
        raise ValueError("resolution_scale must be positive")
    depth: Optional[torch.Tensor] = None
    if frame.depth_path is not None:
        depth_array = np.load(frame.depth_path).astype(np.float32)
        depth_array = np.squeeze(depth_array)
        if depth_array.ndim != 2:
            raise ValueError(f"Expected a 2-D depth map in {frame.depth_path}")
        depth = torch.from_numpy(depth_array.copy()).unsqueeze(0)

    with Image.open(frame.image_path) as source:
        rgb_array = np.asarray(source.convert("RGB"), dtype=np.float32).copy() / 255.0
    image = torch.from_numpy(rgb_array).permute(2, 0, 1).contiguous()
    original_height, original_width = image.shape[-2:]

    # Official iMED NVS RGB frames are 1280x1024 while the metric depth maps
    # are 640x512.  Use the depth grid as the native training grid and resize
    # RGB/masks down to it, matching the official starter loader.  Falling
    # back to RGB resolution keeps the utility usable when depth is absent.
    if depth is not None:
        base_height, base_width = depth.shape[-2:]
    else:
        base_height, base_width = original_height, original_width
    target_height = max(1, int(round(base_height / resolution_scale)))
    target_width = max(1, int(round(base_width / resolution_scale)))
    target_size = (target_height, target_width)
    image = _resize_tensor(image, target_size, "bilinear")
    if depth is not None:
        depth = _resize_tensor(depth, target_size, "nearest")

    if frame.tool_mask_path is not None:
        with Image.open(frame.tool_mask_path) as source:
            tool = np.asarray(source.convert("L"), dtype=np.uint8).copy()
        valid_mask = torch.from_numpy(tool < 128).unsqueeze(0)
        valid_mask = _resize_tensor(valid_mask.float(), target_size, "nearest") > 0.5
    else:
        valid_mask = torch.ones((1, target_height, target_width), dtype=torch.bool)

    if depth is not None:
        valid_mask &= torch.isfinite(depth) & (depth > 0)

    intrinsic = frame.intrinsic.copy()
    intrinsic[0, :] *= target_width / float(original_width)
    intrinsic[1, :] *= target_height / float(original_height)
    return image, depth, valid_mask, intrinsic


def _supported_camera_kwargs(camera_info_type: Type[NamedTuple], values: dict) -> dict:
    fields = getattr(camera_info_type, "_fields", ())
    if not fields:
        raise TypeError("camera_info_type must be a typing.NamedTuple class")
    return {field: values[field] for field in fields if field in values}


def build_imed_camera_splits(
    root: Union[str, Path],
    camera_info_type: Type[NamedTuple],
    focal2fov: Callable[[float, int], float],
    resolution_scale: float = 1.0,
    image_as_pil: bool = False,
) -> Tuple[List[NamedTuple], List[NamedTuple], List[NamedTuple]]:
    """Build native CameraInfo lists for train, novel-view test, and video."""
    sequence_root = Path(root)
    train_frames, test_frames = _discover_frames(sequence_root)

    def convert(frames: Iterable[IMedFrame]) -> List[NamedTuple]:
        cameras: List[NamedTuple] = []
        for frame in frames:
            image, depth, valid_mask, intrinsic = _load_frame_tensors(frame, resolution_scale)
            height, width = image.shape[-2:]
            camera_image = image
            if image_as_pil:
                camera_image = Image.fromarray(
                    (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8),
                    mode="RGB",
                )
            w2c = np.linalg.inv(frame.c2w)
            values = {
                "uid": frame.uid,
                "R": w2c[:3, :3].T.astype(np.float32),
                "T": w2c[:3, 3].astype(np.float32),
                "FovY": focal2fov(float(intrinsic[1, 1]), height),
                "FovX": focal2fov(float(intrinsic[0, 0]), width),
                "image": camera_image,
                "image_path": str(frame.image_path),
                "image_name": f"{frame.endoscope}_{frame.image_path.stem}",
                "width": width,
                "height": height,
                "time": frame.time,
                "fid": frame.time,
                "depth": depth,
                "mask_depth": valid_mask,
                "mask": valid_mask,
                "mono": None,
                "Znear": 1.0e-6,
                "Zfar": 1.0e6,
                "pc": None,
                "principal_point_ndc": np.asarray(
                    [intrinsic[0, 2] / width, intrinsic[1, 2] / height], dtype=np.float32
                ),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2]),
                "depth_path": str(frame.depth_path) if frame.depth_path else "",
                "depth_name": frame.depth_path.stem if frame.depth_path else "",
                "intrinsic": intrinsic,
            }
            cameras.append(camera_info_type(**_supported_camera_kwargs(camera_info_type, values)))
        return cameras

    train_cameras = convert(train_frames)
    test_cameras = convert(test_frames)
    # The video split intentionally follows the held-out physical camera.
    video_cameras = list(test_cameras)
    return train_cameras, test_cameras, video_cameras


def build_imed_dynamic_camera_splits(
    root,
    camera_info_type,
    camera_type,
    focal2fov,
    resolution_scale=1.0,
    style="standard",
):
    """Build the instantiated Camera objects used directly by dynamic trainers."""
    # The repository's CameraInfo often omits depth/mask because its native
    # Endo loader returns instantiated Camera objects. Keep an internal full
    # record here and use camera_info_type only for API symmetry.
    train_infos, test_infos, video_infos = build_imed_camera_splits(
        root,
        IMedCameraInfo,
        focal2fov,
        resolution_scale=resolution_scale,
    )

    def convert(infos):
        cameras = []
        for info in infos:
            common = dict(
                R=info.R,
                T=info.T,
                FoVx=info.FovX,
                FoVy=info.FovY,
                image=info.image,
                depth=info.depth,
                mask=info.mask,
                gt_alpha_mask=None,
                image_name=info.image_name,
                uid=info.uid,
                data_device=torch.device("cuda"),
                time=info.time,
                Znear=1.0e-6,
                Zfar=1.0e6,
            )
            if style == "endo4dgx":
                cameras.append(
                    camera_type(
                        idx=info.uid,
                        prior=info.image,
                        reference=None,
                        illu_type=None,
                        K=info.intrinsic,
                        h=info.height,
                        w=info.width,
                        **common,
                    )
                )
            else:
                constructor_parameters = inspect.signature(camera_type.__init__).parameters
                if "K" in constructor_parameters:
                    common.update(K=info.intrinsic, h=info.height, w=info.width)
                cameras.append(camera_type(colmap_id=info.uid, **common))
        return cameras

    return convert(train_infos), convert(test_infos), convert(video_infos)


def imed_embedding_info() -> Dict:
    """Minimal embedding metadata expected by Endo-4DGX."""
    return {
        "dataset": "iMED_NVS",
        "train_camera": "endoscope2/L",
        "test_camera": "endoscope1/L",
        "num_embeddings": 2,
    }


def imed_nerf_normalization(get_normalization, train_cameras, test_cameras=None, fallback_radius=10.0):
    """Normalize from training cameras only and handle a static-camera radius."""
    cameras = list(train_cameras)
    normalization = get_normalization(cameras)
    radius = float(normalization.get("radius", 0.0))
    if not np.isfinite(radius) or radius < 1.0e-6:
        normalization["radius"] = float(fallback_radius)
    return normalization
