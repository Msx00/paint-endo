#!/usr/bin/env python3
"""Target-free internal metrics on virtual predictions and re-observed pixels."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def ssim(a, b, mask):
    # Global masked SSIM is stable for sparse re-observation and dependency-free.
    values = []
    for channel in range(3):
        x, y = a[..., channel][mask], b[..., channel][mask]
        if not x.size:
            continue
        mx, my = x.mean(), y.mean(); vx, vy = x.var(), y.var()
        cov = ((x - mx) * (y - my)).mean()
        values.append(((2 * mx * my + .01 ** 2) * (2 * cov + .03 ** 2)) /
                      ((mx * mx + my * my + .01 ** 2) * (vx + vy + .03 ** 2)))
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--predictions", required=True, help="frame_NNNNNN.png directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads((Path(args.cache) / "reobs_manifest.json").read_text())
    rows = []
    for row in payload["frames"]:
        path = Path(args.predictions) / "frame_{:06d}.png".format(row["frame_id"])
        if not path.is_file():
            continue
        with np.load(row["cache"], allow_pickle=False) as z:
            target, mask = z["reobs_rgb"], z["reobs_mask"].astype(bool)
            hole = z["virtual_mask"].astype(bool)
            known, warp = ~hole, z["virtual_warp"]
        pred = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if pred.shape[:2] != mask.shape:
            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]))
        mse = float(np.mean((pred[mask] - target[mask]) ** 2)) if mask.any() else float("nan")
        rows.append({"frame_id": row["frame_id"], "reobs_l1": float(np.mean(np.abs(pred[mask] - target[mask]))),
            "reobs_psnr": float(-10 * np.log10(mse + 1e-8)), "reobs_ssim": ssim(pred, target, mask),
            "reobs_coverage": float(mask.sum() / max(1, hole.sum())),
            "stereo_agreement": row.get("stereo_agreement", 0.0),
            "known_region_drift": float(np.mean(np.abs(pred[known] - warp[known])))})
    if not rows:
        raise RuntimeError("no matched predictions")
    keys = [key for key in rows[0] if key != "frame_id"]
    summary = {key: float(np.nanmean([row[key] for row in rows])) for key in keys}
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "internal_metrics.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
