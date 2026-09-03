#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCENE="$(cd "${1:?Usage: run_scene.sh SCENE OUTPUT [GPU_ID]}" && pwd -P)"
OUTPUT_ARG="${2:?Missing output directory}"
GPU_ID="${3:-0}"
mkdir -p "$OUTPUT_ARG"
OUTPUT="$(cd "$OUTPUT_ARG" && pwd -P)"
PYTHON="${GTCOM_PYTHON:-/home/ma_sx/miniconda3/envs/foundation_stereo/bin/python3.11}"
SCENE_NAME="$(basename "$SCENE")"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

EXTRA_PREPARE=()
EXTRA_INFER=()
if [[ -n "${MAX_FRAMES:-}" ]]; then
  EXTRA_PREPARE+=(--max-frames "$MAX_FRAMES")
  EXTRA_INFER+=(--max-frames "$MAX_FRAMES")
fi
if [[ -n "${DOMAIN_LORA:-}" ]]; then
  [[ -f "$DOMAIN_LORA/pytorch_lora_weights.safetensors" ]] || {
    echo "Missing fine-tuned LoRA weights: $DOMAIN_LORA/pytorch_lora_weights.safetensors" >&2
    exit 1
  }
  EXTRA_INFER+=(
    --domain-lora "$DOMAIN_LORA"
    --domain-lora-scale "${DOMAIN_LORA_SCALE:-0.8}"
  )
fi
if [[ "${INPAINT_EXTERIOR_ONLY:-1}" == "1" ]]; then
  EXTRA_INFER+=(--inpaint-exterior-only)
fi
if [[ "${OVERWRITE:-0}" == "1" || "${PREPARE_OVERWRITE:-0}" == "1" ]]; then
  EXTRA_PREPARE+=(--overwrite)
fi
if [[ "${OVERWRITE:-0}" == "1" || "${INFER_OVERWRITE:-0}" == "1" ]]; then
  EXTRA_INFER+=(--overwrite)
fi

if [[ -n "${REUSE_WARP_ROOT:-}" ]]; then
  REUSE_ROOT="$(cd "$REUSE_WARP_ROOT" && pwd -P)"
  WARP_DIR="$REUSE_ROOT/$SCENE_NAME/warps"
  [[ -f "$WARP_DIR/warp_manifest.json" ]] || {
    echo "Missing reusable warp manifest: $WARP_DIR/warp_manifest.json" >&2
    exit 1
  }
  echo "Reusing prepared warp: $WARP_DIR"
else
  WARP_DIR="$OUTPUT/warps"
  "$PYTHON" "$ROOT/prepare_scene.py" \
    --scene "$SCENE" --output "$WARP_DIR" \
    --height "${HEIGHT:-512}" --width "${WIDTH:-640}" \
    --confidence-threshold "${CONFIDENCE_THRESHOLD:-0.2}" \
    --seam-kernel "${SEAM_KERNEL:-7}" "${EXTRA_PREPARE[@]}"
fi

"$PYTHON" "$ROOT/infer_lcm.py" \
  --input "$WARP_DIR" --output "$OUTPUT/lcm" \
  --renders-dir "$OUTPUT/renders" \
  --model "${SD15_INPAINT_MODEL:-$ROOT/models/sd15-inpainting}" \
  --lcm-lora "${LCM_LORA_MODEL:-$ROOT/models/lcm-lora-sdv1-5}" \
  --steps "${LCM_STEPS:-4}" --guidance-scale "${GUIDANCE_SCALE:-1.5}" \
  --batch-size "${BATCH_SIZE:-1}" --seed "${SEED:-6666}" \
  --prompt "${PROMPT:-surgical endoscopy image, realistic tissue}" \
  "${EXTRA_INFER[@]}"
