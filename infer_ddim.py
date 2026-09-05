#!/usr/bin/env python3
"""Standard DDIM SD1.5 inpainting with domain LoRA and hard composition."""

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting, DDIMScheduler
from PIL import Image


def write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_pipeline(args):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    pipeline = AutoPipelineForInpainting.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=args.local_files_only,
        variant="fp16", use_safetensors=True, safety_checker=None,
        requires_safety_checker=False,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(
        args.domain_lora, weight_name=args.domain_lora_weight_name,
        adapter_name="endoscopy", local_files_only=True,
    )
    pipeline.set_adapters(["endoscopy"], adapter_weights=[float(args.domain_lora_scale)])
    active = (pipeline.get_active_adapters()
              if hasattr(pipeline, "get_active_adapters") else ["endoscopy"])
    assert not any("lcm" in str(name).lower() for name in active), (
        "DDIM must not enable an LCM adapter: {}".format(active))
    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=args.quiet)
    pipeline.unet.to(memory_format=torch.channels_last)
    if hasattr(pipeline, "vae"):
        pipeline.vae.enable_slicing()
    torch.backends.cuda.matmul.allow_tf32 = True
    return pipeline


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def keep_border_connected_white(mask_image):
    """Mirror infer_lcm.py: retain white components touching the boundary."""
    mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 127
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return Image.fromarray(mask.astype(np.uint8) * 255), np.zeros_like(mask)
    border_labels = np.unique(np.concatenate((
        labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
    )))
    border_labels = border_labels[border_labels != 0]
    exterior = np.isin(labels, border_labels)
    excluded_internal = mask & ~exterior
    return Image.fromarray(exterior.astype(np.uint8) * 255), excluded_internal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--renders-dir", default="")
    parser.add_argument("--model", default="models/sd15-inpainting")
    parser.add_argument("--domain-lora", required=True)
    parser.add_argument("--domain-lora-weight-name", default="pytorch_lora_weights.safetensors")
    parser.add_argument("--domain-lora-scale", type=float, default=0.8)
    parser.add_argument("--prompt", default="surgical endoscopy image, realistic tissue")
    parser.add_argument("--negative-prompt", default="cartoon, illustration, text, watermark")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable-hard-composition", action="store_true")
    parser.add_argument("--inpaint-exterior-only", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if "lcm" in Path(args.domain_lora).name.lower():
        raise ValueError("--domain-lora appears to be an LCM-LoRA; refusing DDIM run")
    args.local_files_only = not args.allow_download

    input_root, output = Path(args.input).resolve(), Path(args.output).resolve()
    renders_dir = Path(args.renders_dir).resolve() if args.renders_dir else None
    manifest = json.loads((input_root / "warp_manifest.json").read_text())
    if not manifest.get("completed", True):
        raise RuntimeError("Warp preparation is incomplete")
    frames = manifest["frames"][:args.max_frames or None]
    if not frames:
        raise RuntimeError("warp_manifest.json contains no frames")
    raw_dir, final_dir = output / "diffusion_raw", output / "final"
    effective_mask_dir = output / "effective_inpaint_mask"
    for directory in (raw_dir, final_dir, effective_mask_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if renders_dir is not None:
        renders_dir.mkdir(parents=True, exist_ok=True)

    inference_settings = {
        "sampler": "ddim", "steps": int(args.steps),
        "guidance_scale": float(args.guidance_scale), "prompt": args.prompt,
        "negative_prompt": args.negative_prompt, "seed": int(args.seed),
        "model": str(Path(args.model).resolve()),
        "domain_lora": str(Path(args.domain_lora).resolve()),
        "domain_lora_scale": float(args.domain_lora_scale),
        "LCM_LORA_USED": False,
        "inpaint_exterior_only": bool(args.inpaint_exterior_only),
        "hard_composition": not args.disable_hard_composition,
        "renders_dir": str(renders_dir) if renders_dir is not None else "",
    }
    prior_manifest = output / "inference_manifest.json"
    prior_results = {}
    if prior_manifest.is_file() and not args.overwrite:
        prior = json.loads(prior_manifest.read_text())
        mismatches = [key for key, value in inference_settings.items()
                      if prior.get(key, value if key == "renders_dir" else None) != value]
        if mismatches:
            raise RuntimeError("Existing inference uses different {}".format(", ".join(mismatches)))
        prior_results = {int(row["frame_id"]): row for row in prior.get("frames", [])}

    results, pending = [], []
    for row in frames:
        identifier = int(row["frame_id"])
        final_path = final_dir / "frame_{:06d}.png".format(identifier)
        if final_path.is_file() and not args.overwrite:
            if renders_dir is not None and not (renders_dir / final_path.name).is_file():
                shutil.copy2(final_path, renders_dir / final_path.name)
            results.append(prior_results.get(identifier, {
                "frame_id": identifier, "output": str(final_path.resolve()),
                "raw_diffusion": str((raw_dir / final_path.name).resolve()),
                "seconds_per_frame_batch": None, "resumed_from_existing_output": True,
            }))
        else:
            pending.append(row)

    def report_payload(elapsed_seconds, completed):
        return dict(inference_settings, **{
            "method": "SD1.5 Inpainting + domain LoRA + DDIM",
            "strict_known_pixel_preservation": not args.disable_hard_composition,
            "E1_RGB_READ": False, "completed": completed,
            "elapsed_seconds": elapsed_seconds,
            "frames": sorted(results, key=lambda item: int(item["frame_id"])),
        })

    if not pending:
        write_json_atomic(prior_manifest, report_payload(0.0, True))
        print("All requested frames are already complete at {}".format(final_dir))
        return
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for practical DDIM inference")
    pipeline = load_pipeline(args)
    started = time.perf_counter()
    for batch in chunks(pending, max(1, args.batch_size)):
        images = [Image.open(row["warped_rgb"]).convert("RGB") for row in batch]
        original_masks = [Image.open(row["inpaint_mask"]).convert("L") for row in batch]
        if args.inpaint_exterior_only:
            filtered = [keep_border_connected_white(mask) for mask in original_masks]
            masks = [item[0] for item in filtered]
            excluded_internal = [item[1] for item in filtered]
        else:
            masks = original_masks
            excluded_internal = [np.zeros((mask.height, mask.width), dtype=bool)
                                 for mask in masks]
        generators = [torch.Generator(device=args.device).manual_seed(
            args.seed + int(row["frame_id"])) for row in batch]
        tick = time.perf_counter()
        generated = pipeline(
            prompt=[args.prompt] * len(batch), negative_prompt=[args.negative_prompt] * len(batch),
            image=images, mask_image=masks, height=images[0].height, width=images[0].width,
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
            generator=generators,
        ).images
        if args.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(args.device))
        elapsed = time.perf_counter() - tick
        if not isinstance(generated, list):
            generated = [generated]
        for row, source_pil, generated_pil, excluded, effective_mask in zip(
                batch, images, generated, excluded_internal, masks):
            identifier = int(row["frame_id"])
            name = "frame_{:06d}.png".format(identifier)
            effective_mask.save(effective_mask_dir / name)
            generated_pil.save(raw_dir / name)
            source = np.asarray(source_pil, dtype=np.uint8)
            generated_array = np.asarray(generated_pil.convert("RGB"), dtype=np.uint8)
            known = np.asarray(Image.open(row["known_mask"]).convert("L")) > 127
            preserve = known | excluded
            final = (np.where(preserve[..., None], source, generated_array)
                     if not args.disable_hard_composition else generated_array).astype(np.uint8)
            final_path = final_dir / name
            Image.fromarray(final).save(final_path)
            if renders_dir is not None:
                shutil.copy2(final_path, renders_dir / name)
            results.append({
                "frame_id": identifier, "output": str(final_path.resolve()),
                "render": str((renders_dir / name).resolve()) if renders_dir else "",
                "raw_diffusion": str((raw_dir / name).resolve()),
                "effective_inpaint_mask": str((effective_mask_dir / name).resolve()),
                "seconds_per_frame_batch": elapsed / len(batch),
                "known_ratio": float(known.mean()),
                "excluded_internal_mask_ratio": float(excluded.mean()),
            })
            print("frame {} complete ({:.3f}s/frame for batch)".format(
                identifier, elapsed / len(batch)), flush=True)
        write_json_atomic(prior_manifest, report_payload(time.perf_counter() - started, False))
    write_json_atomic(prior_manifest, report_payload(time.perf_counter() - started, True))
    print("Completed {} frames at {}".format(len(results), final_dir))


if __name__ == "__main__":
    main()
