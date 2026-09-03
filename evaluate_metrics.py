#!/usr/bin/env python3
"""Evaluate completions with the native-E1 non-tool/overlap protocol."""
import argparse, csv, json, math, sys, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# TASK2_ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(TASK2_ROOT))
from compute_imed_nvs_fullres_metrics import (  # noqa: E402
    build_overlap_mask, gaussian_window, load_valid_tissue_mask,
)

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--prediction-kind", choices=("final", "diffusion_raw"),
                   default="final",
                   help="Evaluate submitted composites (final) or raw diffusion images")
    return p.parse_args()

def load_sample(item):
    scene, pred_path, gt_path, tool_path, overlap = item
    with Image.open(gt_path) as image:
        gt_image = image.convert("RGB"); size = gt_image.size
        gt = np.asarray(gt_image, dtype=np.float32).copy() / 255.0
    with Image.open(pred_path) as image:
        pred_image = image.convert("RGB")
        if pred_image.size != size:
            pred_image = pred_image.resize(size, Image.Resampling.BILINEAR)
        pred = np.asarray(pred_image, dtype=np.float32).copy() / 255.0
    valid = load_valid_tissue_mask(tool_path, size) * overlap
    return (torch.from_numpy(pred).permute(2,0,1).contiguous(),
            torch.from_numpy(gt).permute(2,0,1).contiguous(), valid.contiguous())

def batch_masked_ssim(pred, gt, mask, eps=1e-8):
    channels, window_size = pred.shape[1], 11
    window = gaussian_window(channels, window_size, 1.5, pred.device, pred.dtype)
    pad = window_size // 2
    conv = lambda x: F.conv2d(x, window, padding=pad, groups=channels)
    mu1, mu2 = conv(pred), conv(gt)
    mu1_sq, mu2_sq, mu12 = mu1.square(), mu2.square(), mu1 * mu2
    sigma1_sq, sigma2_sq = conv(pred.square())-mu1_sq, conv(gt.square())-mu2_sq
    sigma12 = conv(pred*gt)-mu12
    values = ((2*mu12+0.01**2)*(2*sigma12+0.03**2)) / (
        (mu1_sq+mu2_sq+0.01**2)*(sigma1_sq+sigma2_sq+0.03**2)+eps)
    mask3 = mask.expand_as(values)
    return (values*mask3).flatten(1).sum(1)/mask3.flatten(1).sum(1).clamp_min(1)

def summarize(rows):
    result = {"frames": len(rows)}
    for key in ("psnr","ssim","lpips"):
        values=np.asarray([r[key] for r in rows]); values=values[np.isfinite(values)]
        result[key]=float(values.mean()) if values.size else None
        result[key+"_std"]=float(values.std()) if values.size else None
    return result

def main():
    args=arguments(); outputs=args.outputs.resolve(); data=args.data_root.resolve()
    report=args.report_dir.resolve(); report.mkdir(parents=True, exist_ok=True)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
                        else "cpu" if args.device=="auto" else args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    paths=sorted(outputs.glob("*/lcm/{}/frame_*.png".format(args.prediction_kind)))
    if not paths:
        raise RuntimeError("No <scene>/lcm/{}/frame_*.png outputs found".format(
            args.prediction_kind))
    overlaps, samples, missing = {}, [], []
    for pred in paths:
        scene=pred.parents[2].name; root=data/scene
        gt=root/"endoscope1/L"/pred.name; tool=root/"endoscope1/toolL"/pred.name
        if not gt.is_file(): missing.append(str(gt)); continue
        # Match load_valid_tissue_mask semantics: a missing annotation means
        # there is no tool exclusion mask for this frame (all pixels remain
        # eligible before applying the calibrated overlap mask).
        if not tool.is_file(): tool=None
        with Image.open(gt) as image: size=image.size
        key=(scene,size)
        if key not in overlaps: overlaps[key]=build_overlap_mask(root,size)
        samples.append((scene,pred,gt,tool,overlaps[key]))
    if not samples: raise RuntimeError("No prediction/ground-truth pairs found")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="You are using `torch.load`")
        lpips=LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=False, reduction="none").to(device).eval()
    rows=[]; batch=max(1,args.batch_size)
    pool=ThreadPoolExecutor(max_workers=max(1,args.workers))
    with torch.inference_mode(), pool:
        for start in range(0,len(samples),batch):
            current=samples[start:start+batch]; loaded=list(pool.map(load_sample,current))
            pred=torch.stack([x[0] for x in loaded]).to(device)
            gt=torch.stack([x[1] for x in loaded]).to(device)
            valid=torch.stack([x[2] for x in loaded]).to(device); mask3=valid.expand_as(pred)
            denom=mask3.flatten(1).sum(1).clamp_min(1)
            mse=((pred-gt).square()*mask3).flatten(1).sum(1)/denom
            psnr=-10*torch.log10(mse+1e-8); ssim=batch_masked_ssim(pred,gt,valid)
            lpips_values=lpips(pred*mask3,gt*mask3).reshape(-1)
            for i,(scene,pred_path,gt_path,tool_path,_) in enumerate(current):
                rows.append({"scene":scene,"frame":pred_path.name,"prediction":str(pred_path),
                  "ground_truth":str(gt_path),
                  "tool_mask":str(tool_path) if tool_path is not None else "",
                  "valid_pixels":int(valid[i].sum().cpu()),"psnr":float(psnr[i].cpu()),
                  "ssim":float(ssim[i].cpu()),"lpips":float(lpips_values[i].cpu())})
            print("evaluated {}/{}".format(min(start+len(current),len(samples)),len(samples)),flush=True)
    with (report/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    by_scene=defaultdict(list)
    for row in rows: by_scene[row["scene"]].append(row)
    scenes={name:summarize(values) for name,values in sorted(by_scene.items())}
    overall={"scenes":len(scenes),"frames":len(rows)}
    for key in ("psnr","ssim","lpips"):
        values=[v[key] for v in scenes.values() if v[key] is not None]
        overall[key]=float(np.mean(values)) if values else None
    summary={"protocol":"native E1; Endo1 non-tool AND calibrated E2-to-E1 overlap",
      "prediction_kind":args.prediction_kind,
      "resize":"prediction bilinear-upsampled to native Endo1 GT resolution",
      "lpips":"AlexNet; invalid pixels blacked; normalize=False",
      "aggregation":"frame mean, then sequence macro-mean","device":str(device),
      "matched_frames":len(rows),"missing_ground_truth_frames":len(missing),
      "frames_with_tool_mask":sum(bool(row["tool_mask"]) for row in rows),
      "frames_without_tool_mask":sum(not bool(row["tool_mask"]) for row in rows),
      "overall":overall,"scenes":scenes}
    (report/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (report/"missing_ground_truth.txt").write_text("\n".join(missing)+("\n" if missing else ""))
    print("\nPSNR: {:.6f} dB\nSSIM: {:.6f}\nLPIPS: {:.6f}\nReport: {}".format(
      overall["psnr"],overall["ssim"],overall["lpips"],report/"summary.json"))

if __name__=="__main__": main()
