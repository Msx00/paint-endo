#!/usr/bin/env python3
"""Audit ReObsDiff manifests, dataset output, caches and logs for E1 leakage."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reobsdiff.datasets import ReObsDataset
from reobsdiff.leakage import assert_no_target_view, audit_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--training-output", default="")
    args = parser.parse_args()
    root = Path(args.cache).resolve()
    direct = root / "reobs_manifest.json"
    manifests = [direct] if direct.is_file() else sorted(root.glob("*/reobs_manifest.json"))
    if not manifests:
        raise FileNotFoundError("no ReObsDiff manifests below {}".format(root))
    frame_count = 0
    for manifest_path in manifests:
        manifest = audit_json(manifest_path)
        if not manifest.get("completed"):
            raise RuntimeError("incomplete cache manifest: {}".format(manifest_path))
        if manifest.get("target_view_rgb_reads") != 0 or manifest.get("target_view_depth_reads") != 0:
            raise RuntimeError("nonzero target-view read count in {}".format(manifest_path))
        for row in manifest.get("frames", []):
            assert_no_target_view(row, "manifest frame")
            with np.load(row["cache"], allow_pickle=False) as value:
                assert_no_target_view(value.files, "cache array keys")
            frame_count += 1
    dataset = ReObsDataset(root)
    sample = dataset[0]
    assert_no_target_view(sample, "dataset __getitem__")
    if args.training_output:
        output = Path(args.training_output)
        config = audit_json(output / "training_config.json")
        if config.get("target_view_rgb_reads") != 0 or config.get("target_view_depth_reads") != 0:
            raise RuntimeError("nonzero target-view read count in training config")
        log = output / "training_log.jsonl"
        if log.is_file():
            for number, line in enumerate(log.read_text().splitlines(), 1):
                row = json.loads(line); assert_no_target_view(row, "training log line {}".format(number))
                if row.get("target_view_rgb_reads") != 0 or row.get("target_view_depth_reads") != 0:
                    raise RuntimeError("nonzero target read count at log line {}".format(number))
    print("TARGET_VIEW_LEAKAGE_CHECK: PASS scenes={} frames={}".format(
        len(manifests), frame_count))


if __name__ == "__main__":
    main()
