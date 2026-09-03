"""Datasets for self-supervised and Endo1-supervised LoRA adaptation."""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def discover_training_images(data_root):
    return sorted(Path(data_root).glob("session_*/endoscope2/L/frame_*.png"))


def discover_masks(mask_root):
    if not mask_root:
        return []
    return sorted(Path(mask_root).glob("**/inpaint_mask/frame_*.png"))


def irregular_mask(height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    target = random.uniform(0.35, 0.70) * height * width
    attempts = 0
    while np.count_nonzero(mask) < target and attempts < 48:
        attempts += 1
        center = (random.randrange(width), random.randrange(height))
        axes = (random.randint(max(8, width // 20), max(12, width // 4)),
                random.randint(max(8, height // 20), max(12, height // 4)))
        cv2.ellipse(mask, center, axes, random.randrange(180), 0, 360, 255, -1)
        if np.count_nonzero(mask) >= target:
            break
    if random.random() < 0.75:
        side = random.randrange(4)
        fraction = random.uniform(0.05, 0.25)
        if side == 0:
            mask[:int(height * fraction)] = 255
        elif side == 1:
            mask[-int(height * fraction):] = 255
        elif side == 2:
            mask[:, :int(width * fraction)] = 255
        else:
            mask[:, -int(width * fraction):] = 255
    return mask


class EndoscopyInpaintDataset(Dataset):
    def __init__(self, data_root, mask_root="", height=512, width=640,
                 repeats=1, seed=6666):
        self.images = discover_training_images(data_root)
        self.masks = discover_masks(mask_root)
        self.height, self.width = int(height), int(width)
        self.repeats, self.seed = max(1, int(repeats)), int(seed)
        if not self.images:
            raise RuntimeError("No E2-L training images found at {}".format(data_root))

    def __len__(self):
        return len(self.images) * self.repeats

    def __getitem__(self, index):
        random.seed(self.seed + index + random.randrange(1 << 16))
        path = self.images[index % len(self.images)]
        image = np.asarray(Image.open(path).convert("RGB").resize(
            (self.width, self.height), Image.Resampling.LANCZOS
        ), dtype=np.float32) / 255.0
        if self.masks and random.random() < 0.8:
            mask_path = random.choice(self.masks)
            mask = np.asarray(Image.open(mask_path).convert("L").resize(
                (self.width, self.height), Image.Resampling.NEAREST
            ), dtype=np.uint8)
            # Small translations prevent memorizing a fixed camera boundary.
            transform = np.float32([[1, 0, random.randint(-32, 32)],
                                    [0, 1, random.randint(-24, 24)]])
            mask = cv2.warpAffine(mask, transform, (self.width, self.height),
                                  flags=cv2.INTER_NEAREST, borderValue=255)
        else:
            mask = irregular_mask(self.height, self.width)
        mask = (mask > 127).astype(np.float32)
        pixel_values = torch.from_numpy(image).permute(2, 0, 1) * 2.0 - 1.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return {
            "pixel_values": pixel_values,
            "mask": mask_tensor,
            "masked_pixel_values": pixel_values * (1.0 - mask_tensor),
            "path": str(path),
        }


def read_scene_names(path):
    if not path:
        return None
    names = []
    with Path(path).open("r") as handle:
        for raw_line in handle:
            value = raw_line.split("#", 1)[0].strip()
            if value:
                names.append(value)
    if not names:
        raise RuntimeError("Scene list is empty: {}".format(path))
    if len(names) != len(set(names)):
        raise RuntimeError("Scene list contains duplicate names: {}".format(path))
    return set(names)


class SupervisedEndo1InpaintDataset(Dataset):
    """Projected Endo2 conditions paired with same-frame Endo1-L targets."""

    def __init__(self, data_root, warp_root, scenes_file="", height=512,
                 width=640, repeats=1):
        self.data_root = Path(data_root).resolve()
        self.warp_root = Path(warp_root).resolve()
        self.height, self.width = int(height), int(width)
        self.repeats = max(1, int(repeats))
        selected = read_scene_names(scenes_file)
        self.samples, found_scenes = [], set()
        for manifest_path in sorted(self.warp_root.glob("*/warps/warp_manifest.json")):
            scene = manifest_path.parents[1].name
            if selected is not None and scene not in selected:
                continue
            payload = json.loads(manifest_path.read_text())
            if not payload.get("completed", True):
                raise RuntimeError("Incomplete warp manifest: {}".format(manifest_path))
            if payload.get("depth_source") != "endoscope2/depthL ground truth":
                raise RuntimeError("Not a GT-depth manifest: {}".format(manifest_path))
            found_scenes.add(scene)
            for row in payload.get("frames", []):
                frame_id = int(row["frame_id"])
                target = self.data_root / scene / "endoscope1" / "L" / (
                    "frame_{:06d}.png".format(frame_id)
                )
                condition = Path(row["warped_rgb"])
                mask = Path(row["inpaint_mask"])
                if not target.is_file():
                    raise FileNotFoundError("Missing Endo1-L target {}".format(target))
                if not condition.is_file() or not mask.is_file():
                    raise FileNotFoundError("Missing condition/mask in {}".format(manifest_path))
                tool_mask = self.data_root / scene / "endoscope1" / "toolL" / (
                    "frame_{:06d}.png".format(frame_id)
                )
                self.samples.append((
                    scene, frame_id, condition, mask, target,
                    tool_mask if tool_mask.is_file() else None,
                ))
        if selected is not None and found_scenes != selected:
            missing = sorted(selected.difference(found_scenes))
            raise RuntimeError("No completed warp manifest for scenes {}".format(missing))
        if not self.samples:
            raise RuntimeError("No supervised pairs found below {}".format(self.warp_root))
        self.scenes = sorted(found_scenes)

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, index):
        scene, frame_id, condition_path, mask_path, target_path, tool_path = self.samples[
            index % len(self.samples)
        ]
        size = (self.width, self.height)
        condition = np.asarray(Image.open(condition_path).convert("RGB").resize(
            size, Image.Resampling.LANCZOS
        ), dtype=np.float32).copy() / 255.0
        target = np.asarray(Image.open(target_path).convert("RGB").resize(
            size, Image.Resampling.LANCZOS
        ), dtype=np.float32).copy() / 255.0
        mask = np.asarray(Image.open(mask_path).convert("L").resize(
            size, Image.Resampling.NEAREST
        ), dtype=np.uint8).copy() > 127
        mask_u8 = mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        boundary = cv2.dilate(mask_u8, kernel) != cv2.erode(mask_u8, kernel)
        if tool_path is None:
            tissue = np.ones((self.height, self.width), dtype=np.float32)
        else:
            tool = np.asarray(Image.open(tool_path).convert("L").resize(
                size, Image.Resampling.NEAREST
            ), dtype=np.uint8).copy()
            tissue = (tool < 128).astype(np.float32)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        condition_tensor = torch.from_numpy(condition).permute(2, 0, 1) * 2.0 - 1.0
        target_tensor = torch.from_numpy(target).permute(2, 0, 1) * 2.0 - 1.0
        # Match StableDiffusionInpaintPipeline preprocessing: the conditioning
        # image contains only pixels outside the white inpainting mask.
        masked_condition = condition_tensor * (1.0 - mask_tensor)
        return {
            "pixel_values": target_tensor,
            "mask": mask_tensor,
            "boundary": torch.from_numpy(boundary.astype(np.float32)).unsqueeze(0),
            "tissue": torch.from_numpy(tissue).unsqueeze(0),
            "masked_condition": masked_condition,
            "scene": scene,
            "frame_id": frame_id,
            "has_tool_mask": tool_path is not None,
        }
