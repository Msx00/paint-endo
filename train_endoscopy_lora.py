#!/usr/bin/env python
"""LoRA-adapt SD1.5 Inpainting to E2 endoscopy images without E1 targets."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

from stereocom.training_data import EndoscopyInpaintDataset


def save_lora(unet, accelerator, output):
    model = accelerator.unwrap_model(unet)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
    StableDiffusionInpaintPipeline.save_lora_weights(
        save_directory=str(output), unet_lora_layers=state,
        weight_name="pytorch_lora_weights.safetensors",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mask-root", default="")
    parser.add_argument("--model", default="models/sd15-inpainting")
    parser.add_argument("--output", default="models/endoscopy-lora")
    parser.add_argument("--prompt", default="surgical endoscopy image, realistic tissue")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()
    # Accelerate requires these guards for consumer Ada GPUs; this is a
    # single-GPU job, so disabling multi-GPU P2P/IB has no throughput cost.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        mixed_precision="fp16", gradient_accumulation_steps=args.gradient_accumulation,
        log_with="tensorboard", project_dir=str(output / "logs"),
    )
    torch.manual_seed(args.seed)
    model_path = str(Path(args.model).resolve())
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
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
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", local_files_only=True)
    vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    ))
    unet.enable_gradient_checkpointing()
    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    # Keep the frozen backbone in FP16 while updating LoRA weights in FP32.
    unet.to(dtype=torch.float16)
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, betas=(0.9, 0.999),
                                  weight_decay=1e-2, eps=1e-8)
    dataset = EndoscopyInpaintDataset(
        args.data_root, args.mask_root, args.height, args.width,
        repeats=max(1, math.ceil(args.steps * args.batch_size / 4000)), seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
    tokens = tokenizer(
        [args.prompt], padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(accelerator.device)
    with torch.no_grad():
        prompt_embeds = text_encoder(tokens)[0]
    accelerator.init_trackers("endoscopy_lora", config=vars(args))

    global_step = 0
    while global_step < args.steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                pixels = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
                masked = batch["masked_pixel_values"].to(accelerator.device, dtype=torch.float16)
                mask = batch["mask"].to(accelerator.device, dtype=torch.float16)
                with torch.no_grad():
                    latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                    masked_latents = vae.encode(masked).latent_dist.sample() * vae.config.scaling_factor
                latent_mask = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                          (latents.shape[0],), device=latents.device).long()
                noisy = scheduler.add_noise(latents, noise, timesteps)
                model_input = torch.cat((noisy, latent_mask, masked_latents), dim=1)
                embeddings = prompt_embeds.expand(latents.shape[0], -1, -1)
                prediction = unet(model_input, timesteps, embeddings).sample
                target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(
                    latents, noise, timesteps
                )
                weights = 1.0 + 2.0 * latent_mask
                loss = ((prediction.float() - target.float()).square() * weights).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({"loss": float(loss.detach())}, step=global_step)
                if accelerator.is_main_process and (global_step == 1 or global_step % 25 == 0):
                    print("step {}/{} loss={:.6f}".format(global_step, args.steps, float(loss.detach())), flush=True)
                if accelerator.is_main_process and global_step % args.save_every == 0:
                    save_lora(unet, accelerator, output / "checkpoint-{:06d}".format(global_step))
                if global_step >= args.steps:
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora(unet, accelerator, output)
        (output / "training_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    accelerator.end_training()


if __name__ == "__main__":
    main()
