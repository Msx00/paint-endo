#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OUTPUT_ARG="${1:-$ROOT/output}"
DATA_ARG="${2:-/home/ma_sx/Project/iMed_task2/task2-nvs}"
REPORT_ROOT="${3:-$OUTPUT_ARG/metrics}"
INFERENCE_SUBDIR="${INFERENCE_SUBDIR:-lcm}"
[[ "$INFERENCE_SUBDIR" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "Invalid INFERENCE_SUBDIR: $INFERENCE_SUBDIR" >&2
  exit 1
}

PYTHON="/home/ma_sx/miniconda3/envs/foundation_stereo/bin/python3.11"
if [[ ! -x "$PYTHON" ]]; then
  echo "foundation_stereo Python not found: $PYTHON" >&2
  exit 1
fi

# This Conda environment was moved from another path, so Python's compiled-in
# OpenSSL certificate path is stale.  Point downloads (including the one-time
# AlexNet weights required by LPIPS) at the CA bundle in the active environment.
CA_BUNDLE="$($PYTHON -c 'import certifi; print(certifi.where())')"
if [[ ! -f "$CA_BUNDLE" ]]; then
  echo "CA certificate bundle not found: $CA_BUNDLE" >&2
  exit 1
fi
export SSL_CERT_FILE="$CA_BUNDLE"
export REQUESTS_CA_BUNDLE="$CA_BUNDLE"

if [[ -n "${RESULT_NAME:-}" ]]; then
  RESULT_NAMES=("$RESULT_NAME")
else
  mapfile -t RESULT_NAMES < <(
    find "$OUTPUT_ARG" -type f \
      -path "*/results/*/$INFERENCE_SUBDIR/final/frame_*.png" \
      -printf '%h\n' | awk -F/ '{print $(NF-2)}' | sort -u
  )
fi

if (( ${#RESULT_NAMES[@]} == 0 )); then
  echo "No methods found under $OUTPUT_ARG/<scene>/results/" >&2
  exit 1
fi

for result_name in "${RESULT_NAMES[@]}"; do
  echo "Evaluating method: $result_name"
  "$PYTHON" "$ROOT/evaluate_metrics.py" \
    --outputs "$OUTPUT_ARG" \
    --data-root "$DATA_ARG" \
    --report-dir "$REPORT_ROOT/$result_name" \
    --result-name "$result_name" \
    --inference-subdir "$INFERENCE_SUBDIR" \
    --prediction-kind "${PREDICTION_KIND:-final}" \
    --device "${DEVICE:-cuda:2}" \
    --batch-size "${BATCH_SIZE:-20}" \
    --workers "${NUM_WORKERS:-40}"
done
