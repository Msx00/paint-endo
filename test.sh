#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
OUTPUT_ARG="${1:-$ROOT/output}"
DATA_ARG="${2:-/home/ma_sx/Project/iMed_task2/task2-nvs}"
REPORT_ARG="${3:-$OUTPUT_ARG/metrics}"
PYTHON="${GTCOM_PYTHON:-python3}"
"$PYTHON" "$ROOT/evaluate_metrics.py" \
  --outputs "$OUTPUT_ARG" --data-root "$DATA_ARG" --report-dir "$REPORT_ARG" \
  --prediction-kind "${PREDICTION_KIND:-final}" \
  --device "${DEVICE:-cuda:3}" --batch-size "${BATCH_SIZE:-20}" \
  --workers "${NUM_WORKERS:-40}"
