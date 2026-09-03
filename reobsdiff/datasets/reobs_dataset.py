"""Tensor dataset backed solely by an audited ReObsDiff cache."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from reobsdiff.leakage import assert_no_target_view


class ReObsDataset(Dataset):
    def __init__(self, cache_root, repeats=1):
        self.root = Path(cache_root)
        direct = self.root / "reobs_manifest.json"
        manifest_paths = [direct] if direct.is_file() else sorted(
            self.root.glob("*/reobs_manifest.json"))
        if not manifest_paths:
            raise FileNotFoundError(
                "no reobs_manifest.json directly below {} or its scene directories".format(self.root))
        self.rows, self.scenes = [], []
        for manifest_path in manifest_paths:
            payload = json.loads(manifest_path.read_text())
            assert_no_target_view(payload, str(manifest_path))
            if not payload.get("completed"):
                raise RuntimeError("incomplete ReObsDiff cache: {}".format(manifest_path))
            rows = payload.get("frames", [])
            if not rows:
                raise RuntimeError("empty ReObsDiff cache: {}".format(manifest_path))
            self.rows.extend(rows)
            self.scenes.append(manifest_path.parent.name)
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.rows) * self.repeats

    def __getitem__(self, index):
        row = self.rows[index % len(self.rows)]
        assert_no_target_view(row, "training cache row")
        with np.load(row["cache"], allow_pickle=False) as z:
            arrays = {key: np.asarray(z[key]).copy() for key in z.files}
        result = {}
        image_keys = {"anchor_rgb", "reciprocal_rgb", "virtual_warp", "reobs_rgb"}
        for key, value in arrays.items():
            if key in image_keys:
                result[key] = torch.from_numpy(value).permute(2, 0, 1).float() * 2 - 1
            elif value.ndim == 2:
                result[key] = torch.from_numpy(value.astype(np.float32)).unsqueeze(0)
            else:
                result[key] = torch.from_numpy(value.astype(np.float32))
        result["scene"] = row["scene"]
        result["frame_id"] = int(row["frame_id"])
        assert_no_target_view(result, "dataset __getitem__")
        return result
