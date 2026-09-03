#!/usr/bin/env python
"""Download pinned SD1.5 inpainting and LCM-LoRA snapshots locally."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "sd15-inpainting": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "lcm-lora-sdv1-5": "latent-consistency/lcm-lora-sdv1-5",
}

SD15_REQUIRED = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, repository in MODELS.items():
        destination = root / name
        print("Downloading {} -> {}".format(repository, destination), flush=True)
        snapshot_download(
            repo_id=repository,
            local_dir=str(destination),
            allow_patterns=SD15_REQUIRED if name == "sd15-inpainting" else None,
            max_workers=2,
        )
    print("Models are ready at {}".format(root))


if __name__ == "__main__":
    main()
