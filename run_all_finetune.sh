#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_DATA_ROOT="/home/ma_sx/Project/iMed_task2/task2-nvs"

DATA_ROOT="$(cd "${1:-$DEFAULT_DATA_ROOT}" && pwd -P)"
if [[ -n "${2:-}" ]]; then
  OUTPUT_ARG="$2"
elif [[ -d /output && -w /output ]]; then
  # Challenge Docker bind mount.
  OUTPUT_ARG="/output-reobsdiff-v2-2gpu-checkpoint-006000"
else
  # Host-side fallback: ordinary users normally cannot create /output.
  OUTPUT_ARG="$ROOT/output"
fi
GPU_ID="${3:-1}"
DEFAULT_DOMAIN_LORA="$ROOT/models/reobsdiff-v2-2gpu/checkpoint-006000"
export DOMAIN_LORA="${DOMAIN_LORA:-$DEFAULT_DOMAIN_LORA}"
[[ -f "$DOMAIN_LORA/pytorch_lora_weights.safetensors" ]] || {
  echo "Missing fine-tuned LoRA weights: $DOMAIN_LORA/pytorch_lora_weights.safetensors" >&2
  exit 1
}
mkdir -p "$OUTPUT_ARG"
OUTPUT_ROOT="$(cd "$OUTPUT_ARG" && pwd -P)"

if [[ -z "${RESULT_NAME:-}" ]]; then
  domain_leaf="$(basename "${DOMAIN_LORA%/}")"
  domain_parent="$(basename "$(dirname "${DOMAIN_LORA%/}")")"
  if [[ "$domain_leaf" == checkpoint-* ]]; then
    RESULT_NAME="${domain_parent}__${domain_leaf}"
  else
    RESULT_NAME="$domain_leaf"
  fi
fi
[[ "$RESULT_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "Invalid RESULT_NAME: $RESULT_NAME" >&2
  exit 1
}
export RESULT_NAME
SCENE_GLOB="${SCENE_GLOB:-session_*}"
shopt -s nullglob
scenes=("$DATA_ROOT"/$SCENE_GLOB)
(( ${#scenes[@]} > 0 )) || { echo "No scenes matched" >&2; exit 1; }

echo "Fine-tuned LoRA: $DOMAIN_LORA"
echo "Domain LoRA scale: ${DOMAIN_LORA_SCALE:-0.8}"
echo "Reusable warps: $OUTPUT_ROOT/<scene>/warps"
echo "Model results: $OUTPUT_ROOT/<scene>/results/$RESULT_NAME"

for scene in "${scenes[@]}"; do
  name="$(basename "$scene")"
  echo "[$name] Endo2 GT-depth warp + fine-tuned LCM completion"
  bash "$ROOT/run_scene_finetune.sh" "$scene" "$OUTPUT_ROOT/$name" "$GPU_ID"
  result_dir="$OUTPUT_ROOT/$name/results/$RESULT_NAME"
  render_count=$(find "$result_dir/renders" -maxdepth 1 -type f -name 'frame_*.png' | wc -l)
  (( render_count > 0 )) || {
    echo "No challenge renders written to $result_dir/renders" >&2
    exit 1
  }
  echo "[$name] published $render_count renders to $result_dir/renders"
done
