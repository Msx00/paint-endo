"""Typed configuration with explicit switches for every paper ablation."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class ReObsConfig:
    height: int = 512
    width: int = 640
    depth_source: str = "foundationstereo"
    foundation_cache: str = "foundation_stereo_intrinsic_normalized"
    use_e1_camera_geometry: bool = True
    pose_mode: str = "mixture"
    e1_like_probability: float = 0.70
    candidate_count: int = 16
    retry_scales: tuple = (1.0, 0.5, 0.25)
    translation_mm: float = 5.0
    rotation_deg: float = 2.0
    min_overlap: float = 0.55
    min_hole_ratio: float = 0.05
    max_hole_ratio: float = 0.35
    target_hole_ratio: float = 0.20
    min_reobs_ratio: float = 0.30
    score_reobs: float = 1.0
    score_overlap: float = 0.25
    score_hole: float = 0.5
    temporal_window: int = 2
    temporal_tau: float = 1.0
    depth_consistency_relative: float = 0.03
    depth_consistency_mm: float = 0.0
    reobs_max_photometric_error: float = 0.20
    mask_mode: str = "all_geometry"
    mask_min_component: int = 32
    mask_dilation: int = 3
    use_reciprocal_training: bool = True
    use_reobs_loss: bool = True
    use_stereo_reobs: bool = True
    use_temporal_reobs: bool = True
    use_reobs_confidence: bool = True
    use_known_loss: bool = True
    use_hard_composition: bool = True
    lambda_mask: float = 3.0
    lambda_boundary: float = 2.0
    lambda_reobs: float = 1.0
    lambda_grad: float = 0.1
    lambda_known: float = 0.1
    snr_gamma: float = 5.0
    reobs_max_timestep: int = 800
    x0_latent_clip: float = 10.0
    lora_rank: int = 16
    seed: int = 6666

    def validate(self):
        if self.depth_source not in ("gt", "foundationstereo"):
            raise ValueError("depth_source must be gt or foundationstereo")
        if self.pose_mode not in ("local", "e1_like", "mixture"):
            raise ValueError("pose_mode must be local, e1_like, or mixture")
        if self.pose_mode == "e1_like" and not self.use_e1_camera_geometry:
            raise ValueError("e1_like requires use_e1_camera_geometry=true")
        if not (0 <= self.min_hole_ratio <= self.max_hole_ratio <= 1):
            raise ValueError("invalid hole-ratio range")
        if self.height % 8 or self.width % 8:
            raise ValueError("height and width must be multiples of 8")
        if not (0 < self.reobs_max_timestep <= 1000):
            raise ValueError("reobs_max_timestep must be in [1, 1000]")
        if self.x0_latent_clip <= 0:
            raise ValueError("x0_latent_clip must be positive")
        return self

    def to_dict(self):
        value = asdict(self)
        value["retry_scales"] = list(value["retry_scales"])
        return value


def _coerce(value):
    value = value.strip()
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.startswith("["):
        return tuple(json.loads(value))
    try:
        return float(value) if any(c in value for c in ".eE") else int(value)
    except ValueError:
        return value.strip("'\"")


def load_config(path=None, overrides=None):
    values = {}
    if path:
        for raw in Path(path).read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line and ":" in line:
                key, value = line.split(":", 1)
                values[key.strip().lower()] = _coerce(value)
    for item in overrides or ():
        key, value = item.split("=", 1)
        values[key.strip().lower()] = _coerce(value)
    known = set(ReObsConfig.__dataclass_fields__)
    unknown = set(values).difference(known)
    if unknown:
        raise KeyError("unknown ReObsDiff config keys: {}".format(sorted(unknown)))
    return ReObsConfig(**values).validate()
