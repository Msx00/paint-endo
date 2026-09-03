#!/usr/bin/env python3
"""Supervise an SD1.5 Inpainting LoRA with same-frame Endo1-L ground truth."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from stereocom.training_data import SupervisedEndo1InpaintDataset


def save_lora(unet, accelerator, output):
    output.mkdir(parents=True, exist_ok=True)
    model = accelerator.unwrap_model(unet)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
    StableDiffusionInpaintPipeline.save_lora_weights(
        save_directory=str(output), unet_lora_layers=state,
        weight_name="pytorch_lora_weights.safetensors",
    )


def compute_snr(scheduler, timesteps):
    alpha = scheduler.alphas_cumprod.to(timesteps.device)[timesteps].float()
    return alpha / (1.0 - alpha).clamp_min(1.0e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="Root containing session_*/endoscope1/L")
    parser.add_argument("--warp-root", required=True,
                        help="Root containing <scene>/warps/warp_manifest.json")
    parser.add_argument("--scenes-file", default="",
                        help="Optional one-scene-name-per-line training split")
    parser.add_argument("--model", default="models/sd15-inpainting")
    parser.add_argument("--output", default="models/endo1-supervised-lora")
    parser.add_argument("--prompt", default="surgical endoscopy image, realistic tissue")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--mask-weight", type=float, default=3.0)
    parser.add_argument("--boundary-weight", type=float, default=2.0)
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--include-tools", action="store_true",
                        help="Do not mask Endo1 toolL pixels out of the loss")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()
    if args.height % 8 or args.width % 8:
        raise ValueError("height and width must be multiples of 8")
    if min(args.steps, args.batch_size, args.gradient_accumulation, args.rank) <= 0:
        raise ValueError("steps, batch size, accumulation and rank must be positive")

    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.gradient_accumulation,
        log_with="tensorboard", project_dir=str(output / "logs"),
    )
    # Use a rank-specific RNG stream so DDP workers do not train with
    # identical diffusion noise and timesteps.
    set_seed(args.seed, device_specific=True)
    model_path = str(Path(args.model).resolve())
    tokenizer = CLIPTokenizer.from_pretrained(
        model_path, subfolder="tokenizer", local_files_only=True
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_path, subfolder="text_encoder", variant="fp16",
        torch_dtype=torch.float16, local_files_only=True,
    ).to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", variant="fp16",
        torch_dtype=torch.float16, local_files_only=True,
    ).to(accelerator.device)
    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", variant="fp16", local_files_only=True,
    )
    scheduler = DDPMScheduler.from_pretrained(
        model_path, subfolder="scheduler", local_files_only=True
    )
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval(); text_encoder.eval()
    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    unet.enable_gradient_checkpointing()
    unet.to(dtype=torch.float16)
    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, betas=(0.9, 0.999),
        weight_decay=1e-2, eps=1e-8,
    )

    base_dataset = SupervisedEndo1InpaintDataset(
        args.data_root, args.warp_root, args.scenes_file,
        args.height, args.width, repeats=1,
    )
    effective_batch = (
        args.batch_size * args.gradient_accumulation * accelerator.num_processes
    )
    repeats = max(1, math.ceil(args.steps * effective_batch / len(base_dataset)))
    dataset = SupervisedEndo1InpaintDataset(
        args.data_root, args.warp_root, args.scenes_file,
        args.height, args.width, repeats=repeats,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
    token_ids = tokenizer(
        [args.prompt], padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(accelerator.device)
    with torch.no_grad():
        prompt_embeds = text_encoder(token_ids)[0]
    config = vars(args).copy()
    config.update({"training_scenes": base_dataset.scenes,
                   "training_pairs": len(base_dataset),
                   "num_processes": accelerator.num_processes,
                   "global_batch_size": effective_batch})
    tracker_config = config.copy()
    tracker_config["training_scenes"] = ",".join(base_dataset.scenes)
    accelerator.init_trackers("endo1_supervised_lora", config=tracker_config)
    if accelerator.is_main_process:
        (output / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
        print("Training {} pairs from {} scenes".format(
            len(base_dataset), len(base_dataset.scenes)
        ), flush=True)

    global_step = 0
    while global_step < args.steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                target_pixels = batch["pixel_values"].to(
                    accelerator.device, dtype=torch.float16
                )
                condition_pixels = batch["masked_condition"].to(
                    accelerator.device, dtype=torch.float16
                )
                mask = batch["mask"].to(accelerator.device, dtype=torch.float16)
                boundary = batch["boundary"].to(accelerator.device, dtype=torch.float16)
                tissue = batch["tissue"].to(accelerator.device, dtype=torch.float16)
                with torch.no_grad():
                    target_latents = vae.encode(target_pixels).latent_dist.sample()
                    target_latents *= vae.config.scaling_factor
                    condition_latents = vae.encode(condition_pixels).latent_dist.sample()
                    condition_latents *= vae.config.scaling_factor
                latent_mask = F.interpolate(mask, target_latents.shape[-2:], mode="nearest")
                latent_boundary = F.interpolate(
                    boundary, target_latents.shape[-2:], mode="nearest"
                )
                latent_tissue = F.interpolate(
                    tissue, target_latents.shape[-2:], mode="nearest"
                )
                if args.include_tools:
                    latent_tissue = torch.ones_like(latent_tissue)
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],), device=target_latents.device,
                ).long()
                noisy_latents = scheduler.add_noise(target_latents, noise, timesteps)
                model_input = torch.cat(
                    (noisy_latents, latent_mask, condition_latents), dim=1
                )
                embeddings = prompt_embeds.expand(target_latents.shape[0], -1, -1)
                prediction = unet(model_input, timesteps, embeddings).sample
                target = noise if scheduler.config.prediction_type == "epsilon" else (
                    scheduler.get_velocity(target_latents, noise, timesteps)
                )
                spatial_weight = (
                    1.0 + args.mask_weight * latent_mask
                    + args.boundary_weight * latent_boundary
                )
                loss_weight = spatial_weight.float() * latent_tissue.float()
                weighted_error = (
                    (prediction.float() - target.float()).square() * loss_weight
                )
                denominator = (
                    loss_weight.flatten(1).sum(1) * prediction.shape[1]
                ).clamp_min(1.0)
                per_sample = weighted_error.flatten(1).sum(1) / denominator
                if args.snr_gamma > 0:
                    snr = compute_snr(scheduler, timesteps)
                    if scheduler.config.prediction_type == "epsilon":
                        snr_weight = torch.minimum(
                            snr, torch.full_like(snr, args.snr_gamma)
                        ) / snr.clamp_min(1.0e-8)
                    else:
                        snr_weight = torch.minimum(
                            snr, torch.full_like(snr, args.snr_gamma)
                        ) / (snr + 1.0)
                    per_sample = per_sample * snr_weight
                loss = per_sample.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"loss": float(loss.detach())}, step=global_step)
                if accelerator.is_main_process and (
                    global_step == 1 or global_step % 25 == 0
                ):
                    print("step {}/{} loss={:.6f}".format(
                        global_step, args.steps, float(loss.detach())
                    ), flush=True)
                if accelerator.is_main_process and global_step % args.save_every == 0:
                    save_lora(unet, accelerator, output / "checkpoint-{:06d}".format(global_step))
                if global_step >= args.steps:
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora(unet, accelerator, output)
        print("Saved supervised LoRA to {}".format(output), flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
