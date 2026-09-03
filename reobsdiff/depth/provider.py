"""E2-only depth providers. FoundationStereo is deliberately cache-first."""

from abc import ABC, abstractmethod
from pathlib import Path
import json
import cv2
import numpy as np


class DepthProvider(ABC):
    source = "abstract"

    @abstractmethod
    def get(self, frame_id, target_hw=None):
        """Return depth_mm, valid_depth_mask and confidence arrays."""


def _resize(values, target_hw):
    if target_hw and tuple(values.shape) != tuple(target_hw):
        return cv2.resize(values, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return values


class GTDepthProvider(DepthProvider):
    source = "gt"

    def __init__(self, scene):
        self.root = Path(scene) / "endoscope2" / "depthL"

    def get(self, frame_id, target_hw=None):
        path = self.root / "frame_{:06d}.npy".format(int(frame_id))
        depth = _resize(np.asarray(np.load(path, allow_pickle=False), np.float32), target_hw)
        valid = np.isfinite(depth) & (depth > 0)
        return {"depth": depth, "valid_depth_mask": valid,
                "confidence": valid.astype(np.float32), "source": self.source,
                "path": str(path.resolve())}


class FoundationStereoDepthProvider(DepthProvider):
    """Read precomputed metric depth; never recompute it inside an epoch.

    Supported cache files are ``frame_NNNNNN.npz`` with ``depth_mm`` and
    optional ``valid``/``confidence``, or ``frame_NNNNNN.npy`` metric depth.
    A producer command may be supplied to the cache builder for integration
    with a local FoundationStereo checkout.
    """
    source = "foundationstereo"

    def __init__(self, scene, cache_dir="foundation_stereo_metric_depth"):
        root = Path(cache_dir)
        self.root = root if root.is_absolute() else Path(scene) / root

    def get(self, frame_id, target_hw=None):
        stem = "frame_{:06d}".format(int(frame_id))
        npz, npy = self.root / (stem + ".npz"), self.root / (stem + ".npy")
        if npz.is_file():
            with np.load(npz, allow_pickle=False) as value:
                if "depth_mm" not in value:
                    raise RuntimeError("{} lacks depth_mm".format(npz))
                depth = np.asarray(value["depth_mm"], np.float32)
                valid = np.asarray(value["valid"], bool) if "valid" in value else depth > 0
                confidence = (np.asarray(value["confidence"], np.float32)
                              if "confidence" in value else valid.astype(np.float32))
            path = npz
        elif npy.is_file():
            depth = np.asarray(np.load(npy, allow_pickle=False), np.float32)
            valid, confidence, path = depth > 0, (depth > 0).astype(np.float32), npy
        else:
            raise FileNotFoundError(
                "FoundationStereo metric cache missing {}.[npz|npy]; precompute once before training"
                .format(self.root / stem))
        depth, valid, confidence = (_resize(x, target_hw) for x in (depth, valid.astype(np.uint8), confidence))
        valid = valid.astype(bool) & np.isfinite(depth) & (depth > 0)
        return {"depth": depth, "valid_depth_mask": valid,
                "confidence": np.clip(confidence, 0, 1), "source": self.source,
                "path": str(path.resolve())}


def make_depth_provider(scene, config):
    return (GTDepthProvider(scene) if config.depth_source == "gt" else
            FoundationStereoDepthProvider(scene, config.foundation_cache))
