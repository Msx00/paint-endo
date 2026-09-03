#!/usr/bin/env python
"""
Generate E1 warps and diffusion masks from Endo2 ground-truth depth.

Mask strategy
-------------
1. Warp Endoscope2 RGB/depth into Endoscope1 view.
2. Use geometric valid_mask as the main visibility/coverage cue.
3. Apply a small morphological closing to remove tiny forward-warp holes/cracks.
4. Invert the resulting known region to obtain the true missing region.
5. Slightly dilate the missing region to include a narrow seam around boundaries.

Important
---------
- White in inpaint_mask (255): region to be generated / inpainted.
- Black in inpaint_mask (0): geometrically supported region to preserve.
- Confidence is saved for diagnostics but is NOT used as a hard mask threshold.
- Endoscope1 RGB GT is never read when constructing the mask.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from stereocom.io_utils import (
    discover_e2_depth_frames,
    load_rgb,
    read_intrinsics,
    read_poses,
    resize_intrinsic,
)
from stereocom.warp import warp_e2_to_e1


def save_rgb(path, rgb):
    """Save float RGB image in [0, 1] as uint8 PNG."""
    value = np.clip(
        np.asarray(rgb, dtype=np.float32) * 255.0,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(path),
        cv2.cvtColor(value, cv2.COLOR_RGB2BGR),
    )


def save_mask(path, mask):
    """Save boolean mask as 0/255 PNG."""
    cv2.imwrite(
        str(path),
        np.asarray(mask, dtype=np.uint8) * 255,
    )


def write_json_atomic(path, payload):
    """
    Write JSON atomically.

    This keeps warp_manifest.json usable even if preparation is interrupted.
    """
    path = Path(path)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    temporary.replace(path)


def make_odd_kernel_size(value):
    """Convert kernel size to a positive odd integer."""
    value = max(1, int(value))

    if value % 2 == 0:
        value += 1

    return value


def build_inpaint_mask(
    valid_mask,
    close_kernel_size=3,
    seam_kernel_size=3,
):
    """
    Construct diffusion inpainting mask from geometric projection validity.

    Parameters
    ----------
    valid_mask : HxW bool
        Pixels successfully covered by E2 -> E1 geometric projection.

    close_kernel_size : int
        Small closing kernel used to fill tiny splatting holes and thin cracks.
        Recommended: 3.

    seam_kernel_size : int
        Small dilation kernel applied to the missing region so that diffusion
        can also repair a narrow projection seam.
        Recommended: 3.

    Returns
    -------
    known_mask : HxW bool
        Geometrically supported pixels after small-hole cleanup.

    missing_mask : HxW bool
        Pixels with no geometric support before seam expansion.

    inpaint_mask : HxW bool
        Final diffusion mask.
        True/white means "generate this pixel".
    """

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool,
    )

    # ------------------------------------------------------------
    # 1. Start purely from geometric support.
    # ------------------------------------------------------------
    known_mask = valid_mask.astype(np.uint8)

    # ------------------------------------------------------------
    # 2. Fill very small forward-warp cracks / isolated holes.
    #
    # Example:
    #
    #    █████ █████
    #          ^
    #      tiny 1-2 px crack
    #
    # should usually NOT become a diffusion-generation region.
    # ------------------------------------------------------------
    if close_kernel_size > 1:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                close_kernel_size,
                close_kernel_size,
            ),
        )

        known_mask = cv2.morphologyEx(
            known_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
        )

    known_mask = known_mask > 0

    # ------------------------------------------------------------
    # 3. True geometrically unsupported region.
    # ------------------------------------------------------------
    missing_mask = ~known_mask

    # ------------------------------------------------------------
    # 4. Slightly enlarge the missing region.
    #
    # This lets diffusion repair the narrow seam between:
    #
    #     warped known content | missing content
    #
    # But unlike the previous 7x7 erosion, this expansion is kept
    # deliberately small.
    # ------------------------------------------------------------
    inpaint_mask = missing_mask.astype(np.uint8)

    if seam_kernel_size > 1:
        seam_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                seam_kernel_size,
                seam_kernel_size,
            ),
        )

        inpaint_mask = cv2.dilate(
            inpaint_mask,
            seam_kernel,
            iterations=1,
        )

    inpaint_mask = inpaint_mask > 0

    return (
        known_mask,
        missing_mask,
        inpaint_mask,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--scene",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )

    # ------------------------------------------------------------
    # Kept only for compatibility / diagnostics.
    #
    # IMPORTANT:
    # Confidence no longer determines the binary inpainting region.
    # ------------------------------------------------------------
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help=(
            "Diagnostic threshold only. "
            "It is NOT used to construct the final inpaint mask."
        ),
    )

    parser.add_argument(
        "--close-kernel",
        type=int,
        default=3,
        help=(
            "Morphological closing kernel applied to valid_mask. "
            "Used to remove tiny projection holes. Default: 3."
        ),
    )

    parser.add_argument(
        "--seam-kernel",
        type=int,
        default=3,
        help=(
            "Dilation kernel applied to the missing region. "
            "Controls the narrow seam included in inpainting. Default: 3."
        ),
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    scene = Path(
        args.scene
    ).resolve()

    output = Path(
        args.output
    ).resolve()

    target_hw = (
        int(args.height),
        int(args.width),
    )

    # ------------------------------------------------------------
    # Validate target size.
    # ------------------------------------------------------------
    if (
        min(target_hw) <= 0
        or target_hw[0] % 8
        or target_hw[1] % 8
    ):
        raise ValueError(
            "height and width must be positive multiples of 8"
        )

    # ------------------------------------------------------------
    # Discover Endoscope2 RGB + GT depth pairs.
    # ------------------------------------------------------------
    pairs = discover_e2_depth_frames(
        scene
    )

    if not pairs:
        raise RuntimeError(
            "No Endoscope2 RGB/depth pairs found in {}".format(
                scene
            )
        )

    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]

    # ------------------------------------------------------------
    # Determine original image resolution.
    # ------------------------------------------------------------
    first = cv2.imread(
        str(pairs[0][1]),
        cv2.IMREAD_GRAYSCALE,
    )

    if first is None:
        raise FileNotFoundError(
            pairs[0][1]
        )

    source_hw = tuple(
        first.shape
    )

    # ------------------------------------------------------------
    # Camera intrinsics / poses.
    # ------------------------------------------------------------
    intrinsics = read_intrinsics(
        scene / "K.txt"
    )

    poses = read_poses(
        scene / "pose.txt"
    )

    k1 = resize_intrinsic(
        intrinsics["K1_L"],
        source_hw,
        target_hw,
    )

    k2 = resize_intrinsic(
        intrinsics["K2_L"],
        source_hw,
        target_hw,
    )

    # ------------------------------------------------------------
    # Output directories.
    # ------------------------------------------------------------
    directories = {
        name: output / name
        for name in (
            "warped_rgb",
            "valid_mask",
            "known_mask",
            "missing_mask",
            "confidence",
            "low_confidence_mask",
            "inpaint_mask",
            "depth_mm",
            "depth_vis",
        )
    }

    for directory in directories.values():
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------
    # Mask kernel parameters.
    # ------------------------------------------------------------
    close_kernel_size = make_odd_kernel_size(
        args.close_kernel
    )

    seam_kernel_size = make_odd_kernel_size(
        args.seam_kernel
    )

    # ------------------------------------------------------------
    # Manifest settings.
    #
    # mask_strategy is deliberately recorded so an old manifest produced
    # by the previous confidence+erosion method cannot silently be reused.
    # ------------------------------------------------------------
    manifest_settings = {
        "scene": str(scene),
        "height": target_hw[0],
        "width": target_hw[1],

        "mask_strategy": (
            "valid_mask"
            "->close"
            "->invert"
            "->small_dilation"
        ),

        "close_kernel": close_kernel_size,
        "seam_kernel": seam_kernel_size,

        "confidence_threshold": float(
            args.confidence_threshold
        ),

        "confidence_used_for_hard_mask": False,

        "depth_source":
            "endoscope2/depthL ground truth",

        "depth_unit":
            "millimetres",

        "E1_RGB_READ":
            False,
    }

    # ------------------------------------------------------------
    # Resume support.
    # ------------------------------------------------------------
    previous = {}

    previous_manifest = (
        output / "warp_manifest.json"
    )

    if (
        previous_manifest.is_file()
        and not args.overwrite
    ):
        payload = json.loads(
            previous_manifest.read_text()
        )

        mismatches = [
            key
            for key, value
            in manifest_settings.items()
            if payload.get(key) != value
        ]

        if mismatches:
            raise RuntimeError(
                "Existing warp_manifest.json uses different {}. "
                "Choose a new output directory or pass --overwrite."
                .format(
                    ", ".join(mismatches)
                )
            )

        previous = {
            int(row["frame_id"]): row
            for row in payload.get(
                "frames",
                [],
            )
        }

    records = []

    # ============================================================
    # Process every frame.
    # ============================================================
    for index, (
        frame_id,
        left_path,
        depth_path,
    ) in enumerate(
        pairs,
        1,
    ):
        name = "frame_{:06d}".format(
            frame_id
        )

        final_path = (
            directories["inpaint_mask"]
            / (name + ".png")
        )

        old_row = previous.get(
            frame_id
        )

        # --------------------------------------------------------
        # Resume previously completed frames.
        # --------------------------------------------------------
        old_files_exist = (
            old_row is not None
            and all(
                Path(old_row[key]).is_file()
                for key in (
                    "warped_rgb",
                    "valid_mask",
                    "known_mask",
                    "missing_mask",
                    "confidence",
                    "inpaint_mask",
                    "depth_mm",
                )
                if key in old_row
            )
        )

        required_old_keys = (
            "warped_rgb",
            "valid_mask",
            "known_mask",
            "missing_mask",
            "confidence",
            "inpaint_mask",
            "depth_mm",
        )

        old_files_exist = (
            old_row is not None
            and all(
                key in old_row
                and Path(
                    old_row[key]
                ).is_file()
                for key in required_old_keys
            )
        )

        if (
            old_files_exist
            and not args.overwrite
        ):
            records.append(
                previous[frame_id]
            )

            print(
                "[{}/{}] {} already prepared".format(
                    index,
                    len(pairs),
                    name,
                ),
                flush=True,
            )

            continue

        # --------------------------------------------------------
        # Load Endoscope2 GT depth.
        # --------------------------------------------------------
        depth_mm = np.asarray(
            np.load(
                depth_path,
                allow_pickle=False,
            ),
            dtype=np.float32,
        )

        if depth_mm.ndim != 2:
            raise RuntimeError(
                "Ground-truth depth must be HxW: {}".format(
                    depth_path
                )
            )

        # --------------------------------------------------------
        # Resize depth using nearest-neighbour to avoid inventing
        # interpolated depth around discontinuities.
        # --------------------------------------------------------
        if depth_mm.shape != target_hw:
            depth_mm = cv2.resize(
                depth_mm,
                (
                    target_hw[1],
                    target_hw[0],
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        depth_valid = (
            np.isfinite(depth_mm)
            & (depth_mm > 0)
        )

        if not depth_valid.any():
            raise RuntimeError(
                "Ground-truth depth has no valid pixels: {}".format(
                    depth_path
                )
            )

        # --------------------------------------------------------
        # Load Endoscope2 RGB.
        # --------------------------------------------------------
        rgb = load_rgb(
            left_path,
            target_hw,
        )

        # --------------------------------------------------------
        # E2 -> E1 geometric projection.
        # --------------------------------------------------------
        warped = warp_e2_to_e1(
            rgb,
            depth_mm,
            depth_valid,
            k2,
            k1,
            poses[0],
            poses[1],
        )

        valid_mask = np.asarray(
            warped["valid_mask"],
            dtype=bool,
        )

        confidence = np.asarray(
            warped["confidence"],
            dtype=np.float32,
        )

        # ========================================================
        # NEW MASK GENERATION
        # ========================================================
        known_mask, missing_mask, inpaint_mask = (
            build_inpaint_mask(
                valid_mask=valid_mask,
                close_kernel_size=close_kernel_size,
                seam_kernel_size=seam_kernel_size,
            )
        )

        # --------------------------------------------------------
        # Confidence is diagnostic only.
        #
        # This is useful for checking whether problematic image
        # regions happen to correspond to low projection confidence,
        # WITHOUT automatically forcing them into diffusion.
        # --------------------------------------------------------
        low_confidence_mask = (
            valid_mask
            & (
                confidence
                < args.confidence_threshold
            )
        )

        # ========================================================
        # Save RGB warp.
        # ========================================================
        warped_rgb_path = (
            directories["warped_rgb"]
            / (name + ".png")
        )

        save_rgb(
            warped_rgb_path,
            warped["rgb"],
        )

        # ========================================================
        # Save masks.
        # ========================================================
        valid_mask_path = (
            directories["valid_mask"]
            / (name + ".png")
        )

        known_mask_path = (
            directories["known_mask"]
            / (name + ".png")
        )

        missing_mask_path = (
            directories["missing_mask"]
            / (name + ".png")
        )

        confidence_path = (
            directories["confidence"]
            / (name + ".png")
        )

        low_confidence_path = (
            directories["low_confidence_mask"]
            / (name + ".png")
        )

        inpaint_mask_path = (
            directories["inpaint_mask"]
            / (name + ".png")
        )

        save_mask(
            valid_mask_path,
            valid_mask,
        )

        save_mask(
            known_mask_path,
            known_mask,
        )

        save_mask(
            missing_mask_path,
            missing_mask,
        )

        save_mask(
            low_confidence_path,
            low_confidence_mask,
        )

        save_mask(
            inpaint_mask_path,
            inpaint_mask,
        )

        cv2.imwrite(
            str(confidence_path),
            np.clip(
                confidence * 255.0,
                0,
                255,
            ).astype(np.uint8),
        )

        # ========================================================
        # Save warped depth.
        # ========================================================
        depth_mm_path = (
            directories["depth_mm"]
            / (name + ".npy")
        )

        np.save(
            str(depth_mm_path),
            np.asarray(
                warped["depth_mm"],
                dtype=np.float32,
            ),
        )

        # ========================================================
        # Depth visualization.
        # ========================================================
        depth_vis = np.zeros(
            target_hw,
            dtype=np.uint8,
        )

        warped_depth = np.asarray(
            warped["depth_mm"],
            dtype=np.float32,
        )

        positive = (
            np.isfinite(warped_depth)
            & (warped_depth > 0)
        )

        if positive.any():
            low, high = np.percentile(
                warped_depth[positive],
                [1, 99],
            )

            denominator = max(
                float(high - low),
                1e-6,
            )

            depth_vis[positive] = np.clip(
                (
                    warped_depth[positive]
                    - low
                )
                / denominator
                * 255.0,
                0,
                255,
            ).astype(np.uint8)

        depth_vis_path = (
            directories["depth_vis"]
            / (name + ".png")
        )

        cv2.imwrite(
            str(depth_vis_path),
            depth_vis,
        )

        # ========================================================
        # Statistics.
        # ========================================================
        valid_ratio = float(
            valid_mask.mean()
        )

        known_ratio = float(
            known_mask.mean()
        )

        missing_ratio = float(
            missing_mask.mean()
        )

        inpaint_ratio = float(
            inpaint_mask.mean()
        )

        low_confidence_ratio = float(
            low_confidence_mask.mean()
        )

        # ========================================================
        # Manifest row.
        # ========================================================
        row = {
            "frame_id":
                int(frame_id),

            "source_rgb":
                str(
                    Path(left_path).resolve()
                ),

            "source_depth":
                str(
                    Path(depth_path).resolve()
                ),

            "warped_rgb":
                str(
                    warped_rgb_path.resolve()
                ),

            "valid_mask":
                str(
                    valid_mask_path.resolve()
                ),

            "known_mask":
                str(
                    known_mask_path.resolve()
                ),

            "missing_mask":
                str(
                    missing_mask_path.resolve()
                ),

            "confidence":
                str(
                    confidence_path.resolve()
                ),

            "low_confidence_mask":
                str(
                    low_confidence_path.resolve()
                ),

            "inpaint_mask":
                str(
                    inpaint_mask_path.resolve()
                ),

            "depth_mm":
                str(
                    depth_mm_path.resolve()
                ),

            "depth_vis":
                str(
                    depth_vis_path.resolve()
                ),

            "valid_ratio":
                valid_ratio,

            "known_ratio":
                known_ratio,

            "missing_ratio":
                missing_ratio,

            "inpaint_ratio":
                inpaint_ratio,

            "low_confidence_ratio":
                low_confidence_ratio,

            "projected_points":
                warped["projected_points"],

            "source_depth_valid_ratio":
                float(
                    depth_valid.mean()
                ),
        }

        records.append(
            row
        )

        # --------------------------------------------------------
        # Save intermediate manifest for resume.
        # --------------------------------------------------------
        write_json_atomic(
            previous_manifest,
            dict(
                manifest_settings,
                completed=False,
                frames=records,
            ),
        )

        print(
            (
                "[{}/{}] {} "
                "valid={:.3f} "
                "known={:.3f} "
                "missing={:.3f} "
                "inpaint={:.3f} "
                "low_conf={:.3f}"
            ).format(
                index,
                len(pairs),
                name,
                valid_ratio,
                known_ratio,
                missing_ratio,
                inpaint_ratio,
                low_confidence_ratio,
            ),
            flush=True,
        )

    # ============================================================
    # Final manifest.
    # ============================================================
    manifest = dict(
        manifest_settings,
        completed=True,
        frames=records,
    )

    write_json_atomic(
        previous_manifest,
        manifest,
    )

    print(
        "Prepared {} frames at {}".format(
            len(records),
            output,
        )
    )


if __name__ == "__main__":
    main()