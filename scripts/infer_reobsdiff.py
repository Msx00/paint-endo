#!/usr/bin/env python3
"""ReObsDiff single-frame inference via the unchanged GTCom LCM engine."""

import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reobsdiff.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prepared target-pose warp directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--model", default="models/sd15-inpainting")
    parser.add_argument("--lcm-lora", default="models/lcm-lora-sdv1-5")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--config", default="configs/reobsdiff.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    script = Path(__file__).resolve().parents[1] / "infer_lcm.py"
    command = [sys.executable, str(script), "--input", args.input, "--output", args.output,
               "--model", args.model, "--lcm-lora", args.lcm_lora,
               "--domain-lora", args.lora, "--steps", str(args.steps), "--device", args.device]
    if args.max_frames:
        command += ["--max-frames", str(args.max_frames)]
    if not cfg.use_hard_composition:
        command += ["--disable-hard-composition"]
    # infer_lcm performs the required hard composition and never reads E1 GT.
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
