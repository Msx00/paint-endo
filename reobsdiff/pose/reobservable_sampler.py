"""Bounded SE(3) sampling scored by overlap, holes and re-observation."""

import cv2
import numpy as np
from reobsdiff.geometry.warp import warp_to_pose


def se3_delta(translation, rotation):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = cv2.Rodrigues(np.asarray(rotation, np.float64))[0].astype(np.float32)
    matrix[:3, 3] = np.asarray(translation, np.float32)
    return matrix


class ReobservablePoseSampler:
    def __init__(self, config):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.last_diagnostics = None

    def _mode(self):
        if self.cfg.pose_mode != "mixture":
            return self.cfg.pose_mode
        if not self.cfg.use_e1_camera_geometry:
            return "local"
        return "e1_like" if self.rng.random() < self.cfg.e1_like_probability else "local"

    def sample(self, rgb, depth, K, T_anchor, observations, T_e1=None, valid_depth=None):
        best = None
        best_rejected = None
        tested = 0
        for retry_scale in self.cfg.retry_scales:
            for _ in range(self.cfg.candidate_count):
                tested += 1
                mode = self._mode()
                if mode == "e1_like" and (not self.cfg.use_e1_camera_geometry or T_e1 is None):
                    mode = "local"
                t = self.rng.uniform(-1, 1, 3) * self.cfg.translation_mm * retry_scale
                r = self.rng.uniform(-1, 1, 3) * np.deg2rad(self.cfg.rotation_deg) * retry_scale
                base = np.asarray(T_e1 if mode == "e1_like" else T_anchor, np.float32)
                Tv = base @ se3_delta(t, r)
                anchor = warp_to_pose(rgb, depth, K, T_anchor, K, Tv, valid_depth)
                overlap = float(anchor["valid_mask"].mean())
                hole = 1.0 - overlap
                reobserved = np.zeros_like(anchor["valid_mask"])
                for obs in observations:
                    projected = warp_to_pose(obs["rgb"], obs["depth"], obs["K"], obs["T"], K, Tv,
                                             obs.get("valid_depth_mask"))
                    reobserved |= projected["valid_mask"]
                hole_mask = ~anchor["valid_mask"]
                reobs = float((hole_mask & reobserved).sum() / max(1, hole_mask.sum()))
                valid = (overlap >= self.cfg.min_overlap and
                         self.cfg.min_hole_ratio <= hole <= self.cfg.max_hole_ratio and
                         reobs >= self.cfg.min_reobs_ratio)
                score = (self.cfg.score_reobs * reobs + self.cfg.score_overlap * overlap -
                         self.cfg.score_hole * abs(hole - self.cfg.target_hole_ratio))
                row = {"T_virtual": Tv, "anchor_warp": anchor, "overlap_ratio": overlap,
                       "hole_ratio": hole, "reobs_ratio": reobs, "score": score,
                       "candidate_count": tested, "pose_mode": mode,
                       "translation_delta_mm": float(np.linalg.norm(t)),
                       "rotation_delta_deg": float(np.rad2deg(np.linalg.norm(r)))}
                if valid and (best is None or score > best["score"]):
                    best = row
                if best_rejected is None or score > best_rejected["score"]:
                    best_rejected = row
            if best is not None:
                self.last_diagnostics = best
                return best
        self.last_diagnostics = best_rejected
        return None
