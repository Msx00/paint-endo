"""Differentiable losses used by both training branches."""

import torch
import torch.nn.functional as F


def _weighted_mean(value, mask, confidence=None):
    weight = mask if confidence is None else mask * confidence
    # Accumulate masked image losses in float32. At 512x512, summing an
    # expanded three-channel mask in fp16 can exceed 65504, making both the
    # numerator and denominator inf and producing an inf/inf NaN.
    value = value.float()
    weight = weight.expand_as(value).float()
    # torch follows IEEE semantics (NaN * 0 == NaN).  Do not let an
    # unselected decoded pixel poison a sparse masked loss.
    weighted = torch.where(weight > 0, value * weight, torch.zeros_like(value))
    return weighted.sum() / weight.sum().clamp_min(1e-8)


def charbonnier_loss(prediction, target, mask, confidence=None, eps=1e-6):
    return _weighted_mean(torch.sqrt((prediction - target).square() + eps), mask, confidence)


def _sobel(image):
    kernel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    kernel_y = kernel_x.t()
    channels = image.shape[1]
    kx = kernel_x[None, None].expand(channels, 1, 3, 3)
    ky = kernel_y[None, None].expand(channels, 1, 3, 3)
    return F.conv2d(image, kx, padding=1, groups=channels), F.conv2d(image, ky, padding=1, groups=channels)


def sobel_loss(prediction, target, mask, confidence=None):
    px, py = _sobel(prediction)
    tx, ty = _sobel(target)
    return _weighted_mean((px - tx).abs() + (py - ty).abs(), mask, confidence)


def known_region_loss(prediction, warped, hole_mask):
    return _weighted_mean((prediction - warped).abs(), 1.0 - hole_mask)


def predict_x0(noisy_latents, model_output, timesteps, scheduler):
    alpha = scheduler.alphas_cumprod.to(noisy_latents.device)[timesteps]
    while alpha.ndim < noisy_latents.ndim:
        alpha = alpha.unsqueeze(-1)
    beta = 1 - alpha
    if scheduler.config.prediction_type == "epsilon":
        return (noisy_latents - beta.sqrt() * model_output) / alpha.sqrt().clamp_min(1e-8)
    if scheduler.config.prediction_type == "v_prediction":
        return alpha.sqrt() * noisy_latents - beta.sqrt() * model_output
    if scheduler.config.prediction_type == "sample":
        return model_output
    raise ValueError("unsupported prediction_type {}".format(scheduler.config.prediction_type))


def min_snr_weight(scheduler, timesteps, gamma):
    alpha = scheduler.alphas_cumprod.to(timesteps.device)[timesteps].float()
    snr = alpha / (1 - alpha).clamp_min(1e-8)
    denominator = snr if scheduler.config.prediction_type == "epsilon" else snr + 1
    return torch.minimum(snr, torch.full_like(snr, gamma)) / denominator.clamp_min(1e-8)
