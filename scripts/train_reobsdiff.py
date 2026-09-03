#!/usr/bin/env python3
"""Train SD1.5 Inpainting LoRA without ever loading an Endoscope1 image."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reobsdiff.config import load_config
from reobsdiff.datasets import ReObsDataset
from reobsdiff.diffusion.losses import charbonnier_loss, known_region_loss, min_snr_weight, predict_x0, sobel_loss
from reobsdiff.leakage import assert_no_target_view


def save_lora(unet, accelerator, output):
    output.mkdir(parents=True, exist_ok=True)
    model = accelerator.unwrap_model(unet)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
    StableDiffusionInpaintPipeline.save_lora_weights(
        save_directory=str(output), unet_lora_layers=state,
        weight_name="pytorch_lora_weights.safetensors")


def encode(vae, pixels):
    return vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor


def model_pass(unet, scheduler, clean_latents, condition_latents, mask, embeddings,
               max_timestep=None):
    noise = torch.randn_like(clean_latents)
    timestep_count = scheduler.config.num_train_timesteps
    if max_timestep is not None:
        timestep_count = min(timestep_count, int(max_timestep) + 1)
    timesteps = torch.randint(0, timestep_count,
                              (clean_latents.shape[0],), device=clean_latents.device).long()
    noisy = scheduler.add_noise(clean_latents, noise, timesteps)
    prediction = unet(torch.cat((noisy, mask, condition_latents), 1), timesteps, embeddings).sample
    target = noise if scheduler.config.prediction_type == "epsilon" else scheduler.get_velocity(clean_latents, noise, timesteps)
    return noisy, prediction, target, timesteps


def ssim_loss(prediction, target, window_size=11):
    """Dense SSIM loss for RGB tensors normalized to [-1, 1]."""
    prediction, target = prediction.float(), target.float()
    padding = window_size // 2
    mean_prediction = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mean_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    prediction_variance = F.avg_pool2d(
        prediction.square(), window_size, stride=1, padding=padding) - mean_prediction.square()
    target_variance = F.avg_pool2d(
        target.square(), window_size, stride=1, padding=padding) - mean_target.square()
    covariance = F.avg_pool2d(
        prediction * target, window_size, stride=1, padding=padding
    ) - mean_prediction * mean_target
    # Images use a dynamic range of two after normalization to [-1, 1].
    c1, c2 = (0.01 * 2) ** 2, (0.03 * 2) ** 2
    score = ((2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)) / (
        (mean_prediction.square() + mean_target.square() + c1)
        * (prediction_variance + target_variance + c2)
    ).clamp_min(1e-8)
    return (1 - score.clamp(-1, 1)).mean()


def per_sample_reobs_losses(prediction, target, mask, confidence, gradient_weight):
    """Keep coverage weights sample-specific when batch size is greater than one."""
    rgb, gradient = [], []
    for index in range(prediction.shape[0]):
        sample = slice(index, index + 1)
        rgb.append(charbonnier_loss(
            prediction[sample], target[sample], mask[sample], confidence[sample]))
        gradient.append(sobel_loss(
            prediction[sample], target[sample], mask[sample], confidence[sample]))
    rgb, gradient = torch.stack(rgb), torch.stack(gradient)
    return rgb, gradient, rgb + gradient_weight * gradient


def reobs_coverage_weights(reobs_mask, hole_mask, low, high):
    hole_mask = hole_mask.float()
    # Count only re-observations that lie inside the original geometric hole.
    # The inpainting mask may be dilated and must not be used as denominator.
    reobserved = (reobs_mask.float() * hole_mask).flatten(1).sum(1)
    holes = hole_mask.flatten(1).sum(1).clamp_min(1)
    coverage = (reobserved / holes).clamp(0, 1)
    weight = ((coverage - low) / (high - low)).clamp(0, 1)
    return coverage, weight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="configs/reobsdiff.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--model", default="models/sd15-inpainting")
    parser.add_argument("--output", default="models/reobsdiff-lora")
    parser.add_argument("--prompt", default="surgical endoscopy image, realistic tissue")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--lambda-e2", type=float, default=0.5,
                        help="weight for dense reciprocal E2 reconstruction")
    parser.add_argument("--e2-gradient-weight", type=float, default=0.1)
    parser.add_argument("--e2-boundary-weight", type=float, default=0.5)
    parser.add_argument("--e2-ssim-weight", type=float, default=0.1)
    parser.add_argument("--e2-known-weight", type=float, default=0.05)
    parser.add_argument(
        "--e2-recon-max-timestep", type=int, default=400,
        help="maximum diffusion timestep used for dense E2 x0 reconstruction supervision")
    parser.add_argument("--reobs-coverage-low", type=float, default=0.10,
                        help="coverage at or below this value receives no ReObs loss")
    parser.add_argument("--reobs-coverage-high", type=float, default=0.40,
                        help="coverage at or above this value receives full ReObs loss")
    args = parser.parse_args()
    if any(weight < 0 for weight in (
            args.lambda_e2, args.e2_gradient_weight, args.e2_boundary_weight,
            args.e2_ssim_weight, args.e2_known_weight)):
        parser.error("E2 reconstruction weights must be non-negative")
    if args.e2_recon_max_timestep < 0:
        parser.error("--e2-recon-max-timestep must be >= 0")
    if not 0 <= args.reobs_coverage_low < args.reobs_coverage_high <= 1:
        parser.error("ReObs coverage thresholds must satisfy 0 <= low < high <= 1")
    cfg = load_config(args.config, args.set)
    assert_no_target_view(vars(args), "training arguments")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else "no",
        gradient_accumulation_steps=args.gradient_accumulation,
        log_with="tensorboard", project_dir=str(output / "logs"))
    set_seed(cfg.seed, device_specific=True)
    model_path = str(Path(args.model).resolve())
    dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder",
        variant="fp16", torch_dtype=dtype, local_files_only=True).to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", variant="fp16", torch_dtype=dtype,
                                        local_files_only=True).to(accelerator.device)
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", variant="fp16", local_files_only=True)
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler", local_files_only=True)
    vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)
    vae.eval(); text_encoder.eval()
    unet.add_adapter(LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_rank,
        init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    unet.enable_gradient_checkpointing(); unet.to(dtype=dtype)
    trainable = [p for p in unet.parameters() if p.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)
    base = ReObsDataset(args.cache)
    effective = args.batch_size * args.gradient_accumulation * accelerator.num_processes
    repeats = max(1, math.ceil(args.steps * effective / len(base)))
    loader = DataLoader(ReObsDataset(args.cache, repeats), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
    ids = tokenizer([args.prompt], padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
    with torch.no_grad():
        prompt = text_encoder(ids)[0]
    run_config = {"method": "reobsdiff", "training_cache": str(Path(args.cache).resolve()),
                  "target_view_rgb_reads": 0, "target_view_depth_reads": 0,
                  "steps": args.steps, "global_batch_size": effective,
                  "training_scenes": base.scenes, "training_frames": len(base),
                  "dense_e2_reconstruction": {
                      "weight": args.lambda_e2,
                      "gradient_weight": args.e2_gradient_weight,
                      "boundary_weight": args.e2_boundary_weight,
                      "ssim_weight": args.e2_ssim_weight,
                      "known_weight": args.e2_known_weight,
                      "max_timestep": args.e2_recon_max_timestep,
                  },
                  "reobs_coverage_weighting": {
                      "low": args.reobs_coverage_low,
                      "high": args.reobs_coverage_high,
                  },
                  "config": cfg.to_dict()}
    assert_no_target_view(run_config, "training config")
    if accelerator.is_main_process:
        (output / "training_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    accelerator.init_trackers("reobsdiff", config={"steps": args.steps, "rank": cfg.lora_rank})
    log_path = output / "training_log.jsonl"
    step = 0
    while step < args.steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                anchor = batch["anchor_rgb"].to(accelerator.device, dtype=dtype)
                reciprocal = batch["reciprocal_rgb"].to(accelerator.device, dtype=dtype)
                reciprocal_mask = batch["reciprocal_mask"].to(accelerator.device, dtype=dtype)
                recip_boundary = batch["reciprocal_boundary"].to(accelerator.device, dtype=dtype)
                virtual = batch["virtual_warp"].to(accelerator.device, dtype=dtype)
                virtual_mask = batch["virtual_mask"].to(accelerator.device, dtype=dtype)
                virtual_valid = batch["virtual_valid"].to(accelerator.device, dtype=dtype)
                geometric_hole = 1.0 - virtual_valid
                reobs_rgb = batch["reobs_rgb"].to(accelerator.device, dtype=dtype)
                reobs_mask = batch["reobs_mask"].to(accelerator.device, dtype=dtype)
                reobs_conf = batch["reobs_confidence"].to(accelerator.device, dtype=dtype)
                with torch.no_grad():
                    anchor_z, reciprocal_z = encode(vae, anchor), encode(vae, reciprocal * (1 - reciprocal_mask))
                    virtual_z = encode(vae, virtual)
                    # Match SD inpainting inference: mask pixels before VAE
                    # encoding, rather than erasing already-mixed latents.
                    virtual_condition_z = encode(vae, virtual * (1 - virtual_mask))
                emb = prompt.expand(anchor.shape[0], -1, -1)
                latent_recip_mask = F.interpolate(reciprocal_mask, anchor_z.shape[-2:], mode="nearest")
                latent_boundary = F.interpolate(recip_boundary, anchor_z.shape[-2:], mode="nearest")
                noisy_a, pred_a, noise_target, times_a = model_pass(
                    unet, scheduler, anchor_z, reciprocal_z, latent_recip_mask, emb)
                weight = 1 + cfg.lambda_mask * latent_recip_mask + cfg.lambda_boundary * latent_boundary
                weighted_error = (pred_a.float() - noise_target.float()).square() * weight.float()
                denominator = (weight.float().flatten(1).sum(1) * pred_a.shape[1]).clamp_min(1)
                per_sample = weighted_error.flatten(1).sum(1) / denominator
                if cfg.snr_gamma > 0:
                    per_sample *= min_snr_weight(scheduler, times_a, cfg.snr_gamma)
                loss_diff = per_sample.mean() if cfg.use_reciprocal_training else pred_a.sum() * 0
                if cfg.use_reciprocal_training and args.lambda_e2 > 0:
                    noisy_e2, pred_e2, _, times_e2 = model_pass(
                        unet, scheduler, anchor_z, reciprocal_z, latent_recip_mask, emb,
                        args.e2_recon_max_timestep)
                    e2_x0 = predict_x0(noisy_e2, pred_e2, times_e2, scheduler)
                    e2_x0 = e2_x0.clamp(-cfg.x0_latent_clip, cfg.x0_latent_clip)
                    e2_prediction = vae.decode(
                        (e2_x0 / vae.config.scaling_factor).to(dtype=dtype)
                    ).sample.clamp(-1, 1)
                    e2_hole_mask = reciprocal_mask
                    known_mask = 1.0 - reciprocal_mask
                    loss_e2_rgb_hole = charbonnier_loss(
                        e2_prediction, anchor, e2_hole_mask)
                    loss_e2_grad_hole = sobel_loss(
                        e2_prediction, anchor, e2_hole_mask)
                    loss_e2_boundary = charbonnier_loss(
                        e2_prediction, anchor, recip_boundary)
                    loss_e2_ssim = ssim_loss(e2_prediction, anchor)
                    loss_e2_known = charbonnier_loss(
                        e2_prediction, anchor, known_mask)
                    loss_e2 = (loss_e2_rgb_hole
                               + args.e2_gradient_weight * loss_e2_grad_hole
                               + args.e2_boundary_weight * loss_e2_boundary
                               + args.e2_ssim_weight * loss_e2_ssim)
                    loss_e2 = loss_e2 + args.e2_known_weight * loss_e2_known
                else:
                    times_e2 = times_a.new_full(times_a.shape, -1)
                    loss_e2_rgb_hole = pred_a.sum() * 0
                    loss_e2_grad_hole = pred_a.sum() * 0
                    loss_e2_boundary = pred_a.sum() * 0
                    loss_e2_ssim = pred_a.sum() * 0
                    loss_e2_known = pred_a.sum() * 0
                    loss_e2 = pred_a.sum() * 0
                latent_virtual_mask = F.interpolate(virtual_mask, virtual_z.shape[-2:], mode="nearest")
                noisy_b, pred_b, _, times_b = model_pass(
                    unet, scheduler, virtual_z, virtual_condition_z,
                    latent_virtual_mask, emb, cfg.reobs_max_timestep)
                x0 = predict_x0(noisy_b, pred_b, times_b, scheduler)
                x0 = x0.clamp(-cfg.x0_latent_clip, cfg.x0_latent_clip)
                prediction = vae.decode((x0 / vae.config.scaling_factor).to(dtype=dtype)).sample.clamp(-1, 1)
                loss_rgb_samples, loss_grad_samples, loss_reobs_samples = per_sample_reobs_losses(
                    prediction, reobs_rgb, reobs_mask, reobs_conf, cfg.lambda_grad)
                coverage, coverage_weight = reobs_coverage_weights(
                    reobs_mask, geometric_hole,
                    args.reobs_coverage_low, args.reobs_coverage_high)
                loss_rgb, loss_grad = loss_rgb_samples.mean(), loss_grad_samples.mean()
                loss_reobs = (coverage_weight * loss_reobs_samples).mean()
                if not cfg.use_reobs_loss:
                    loss_reobs = loss_reobs * 0
                loss_known = known_region_loss(prediction, virtual, virtual_mask)
                if not cfg.use_known_loss:
                    loss_known = loss_known * 0
                loss = (loss_diff + args.lambda_e2 * loss_e2
                        + cfg.lambda_reobs * loss_reobs + cfg.lambda_known * loss_known)
                components = {
                    "loss_total": loss, "loss_diff_self": loss_diff,
                    "loss_e2_exact": loss_e2, "loss_e2_rgb_hole": loss_e2_rgb_hole,
                    "loss_e2_grad_hole": loss_e2_grad_hole,
                    "loss_e2_boundary": loss_e2_boundary,
                    "loss_e2_ssim": loss_e2_ssim, "loss_e2_known": loss_e2_known,
                    "loss_reobs_rgb": loss_rgb, "loss_reobs_grad": loss_grad,
                    "loss_reobs_weighted": loss_reobs,
                    "loss_known": loss_known,
                }
                bad = [name for name, value in components.items()
                       if not bool(torch.isfinite(value.detach()).all())]
                if bad:
                    raise FloatingPointError(
                        "non-finite {} at optimizer step {}; diffusion timesteps={}, "
                        "E2 timesteps={}, reobs timesteps={}, "
                        "x0_abs_max={:.6g}, prediction_finite={}".format(
                            ",".join(bad), step + 1, times_a.detach().cpu().tolist(),
                            times_e2.detach().cpu().tolist(), times_b.detach().cpu().tolist(),
                            float(x0.detach().abs().max()),
                            bool(torch.isfinite(prediction.detach()).all())))
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                step += 1
                metrics = {"loss_total": float(loss.detach()), "loss_diff_self": float(loss_diff.detach()),
                    "loss_e2_exact": float(loss_e2.detach()),
                    "loss_e2_rgb_hole": float(loss_e2_rgb_hole.detach()),
                    "loss_e2_grad_hole": float(loss_e2_grad_hole.detach()),
                    "loss_e2_boundary": float(loss_e2_boundary.detach()),
                    "loss_e2_ssim": float(loss_e2_ssim.detach()),
                    "loss_e2_known": float(loss_e2_known.detach()),
                    # Backward-compatible aliases for existing dashboards.
                    "loss_e2_rgb": float(loss_e2_rgb_hole.detach()),
                    "loss_e2_grad": float(loss_e2_grad_hole.detach()),
                    "loss_reobs_rgb": float(loss_rgb.detach()), "loss_reobs_grad": float(loss_grad.detach()),
                    "loss_reobs_weighted": float(loss_reobs.detach()),
                    "loss_known": float(loss_known.detach()),
                    "diffusion_timestep_max": int(times_a.max().detach()),
                    "e2_recon_timestep_max": int(times_e2.max().detach()),
                    "e2_hole_ratio": float(reciprocal_mask.float().mean().detach()),
                    "geometric_hole_ratio": float(geometric_hole.float().mean().detach()),
                    "virtual_inpaint_mask_ratio": float(virtual_mask.float().mean().detach()),
                    "reobs_coverage": float(coverage.mean().detach()),
                    "reobs_coverage_weight": float(coverage_weight.mean().detach()),
                    "reobs_timestep_max": int(times_b.max().detach()),
                    "x0_abs_max": float(x0.detach().abs().max()),
                    "target_view_rgb_reads": 0, "target_view_depth_reads": 0}
                accelerator.log(metrics, step=step)
                if accelerator.is_main_process:
                    with log_path.open("a") as handle:
                        handle.write(json.dumps(dict(step=step, **metrics)) + "\n")
                    if step == 1 or step % 10 == 0:
                        print("step {}/{} total={loss_total:.6f} diff={loss_diff_self:.6f} "
                              "e2={loss_e2_exact:.6f} reobs={loss_reobs_weighted:.6f} "
                              "coverage={reobs_coverage:.3f} weight={reobs_coverage_weight:.3f}".format(
                                  step, args.steps, **metrics), flush=True)
                    if step % args.save_every == 0:
                        save_lora(unet, accelerator, output / "checkpoint-{:06d}".format(step))
                if step >= args.steps:
                    break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora(unet, accelerator, output)
    accelerator.end_training()


if __name__ == "__main__":
    main()
