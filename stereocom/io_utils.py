"""Strict readers for the iMED Task-2 scene format."""

from pathlib import Path

import cv2
import numpy as np


def frame_number(path):
    digits = "".join(character for character in Path(path).stem if character.isdigit())
    if not digits:
        raise ValueError("Cannot extract frame number from {}".format(path))
    return int(digits)


def discover_e2_pairs(scene):
    scene = Path(scene)
    left = {frame_number(path): path for path in (scene / "endoscope2" / "L").glob("frame_*.png")}
    right = {frame_number(path): path for path in (scene / "endoscope2" / "R").glob("frame_*.png")}
    if not left or set(left) != set(right):
        raise RuntimeError("E2 left/right frame ids are empty or differ in {}".format(scene))
    return [(identifier, left[identifier], right[identifier]) for identifier in sorted(left)]


def discover_e2_depth_frames(scene):
    """Return aligned E2-L RGB and GT-depth frames; E2-R is not required."""
    scene = Path(scene)
    left = {
        frame_number(path): path
        for path in (scene / "endoscope2" / "L").glob("frame_*.png")
    }
    depth = {
        frame_number(path): path
        for path in (scene / "endoscope2" / "depthL").glob("frame_*.npy")
    }
    if not left:
        raise RuntimeError("No E2-L RGB frames found in {}".format(scene))
    if set(left) != set(depth):
        missing_depth = sorted(set(left).difference(depth))
        missing_rgb = sorted(set(depth).difference(left))
        raise RuntimeError(
            "E2-L RGB/depth frame ids differ in {}: missing_depth={}, "
            "missing_rgb={}".format(scene, missing_depth[:5], missing_rgb[:5])
        )
    return [(identifier, left[identifier], depth[identifier]) for identifier in sorted(left)]


def read_intrinsics(path):
    matrices, label, rows = {}, None, []

    def commit():
        if label is not None and rows:
            matrix = np.asarray(rows, dtype=np.float32)
            if matrix.shape != (3, 3):
                raise ValueError("{} in {} is not 3x3".format(label, path))
            matrices[label] = matrix

    with Path(path).open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                commit()
                rows[:] = []
                words = line[1:].strip().split()
                label = words[0] if words and words[0].startswith("K") else None
            elif label is not None:
                rows.append([float(value) for value in line.split()])
    commit()
    missing = {"K1_L", "K2_L"}.difference(matrices)
    if missing:
        raise KeyError("{} misses {}".format(path, sorted(missing)))
    return matrices


def quaternion_xyzw_to_matrix(values):
    x, y, z, w = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("zero-length quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def read_poses(path):
    poses = {}
    with Path(path).open("r") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            if len(values) != 8:
                raise ValueError("Invalid pose at {}:{}".format(path, line_number))
            identifier = int(values[0])
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = quaternion_xyzw_to_matrix(values[4:8])
            matrix[:3, 3] = np.asarray(values[1:4], dtype=np.float32)
            poses[identifier] = matrix
    if 0 not in poses or 1 not in poses:
        raise KeyError("pose.txt needs camera 0 (E2-L) and camera 1 (E1-L)")
    return poses


def resize_intrinsic(intrinsic, source_hw, target_hw):
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    result = np.asarray(intrinsic, dtype=np.float32).copy()
    result[0] *= target_w / float(source_w)
    result[1] *= target_h / float(source_h)
    return result


def load_rgb(path, target_hw):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    if tuple(bgr.shape[:2]) != tuple(target_hw):
        bgr = cv2.resize(bgr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
