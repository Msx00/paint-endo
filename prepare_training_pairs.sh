#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
DATA_ROOT="$(cd "${1:?Usage: prepare_training_pairs.sh DATA_ROOT WARP_ROOT [SCENES_FILE]}" && pwd -P)"

WARP_ARG="${2:?Missing WARP_ROOT}"
SCENES_FILE="${3:-}"

PYTHON="${GTCOM_PYTHON:-python3}"

mkdir -p "$WARP_ARG"
WARP_ROOT="$(cd "$WARP_ARG" && pwd -P)"

if [[ -n "$SCENES_FILE" ]]; then
    mapfile -t scene_names < <(
        sed 's/#.*//' "$SCENES_FILE" | awk 'NF {print $1}'
    )
else
    shopt -s nullglob

    scene_paths=(
        "$DATA_ROOT"/session_*
    )

    scene_names=()

    for path in "${scene_paths[@]}"; do
        scene_names+=(
            "$(basename "$path")"
        )
    done
fi

(( ${#scene_names[@]} > 0 )) || {
    echo "No training scenes found" >&2
    exit 1
}

for name in "${scene_names[@]}"; do

    scene="$DATA_ROOT/$name"

    [[ -d "$scene" ]] || {
        echo "Missing scene: $scene" >&2
        exit 1
    }

    echo "[$name] preparing GT-depth projection pairs"

    "$PYTHON" "$ROOT/prepare_scene.py" \
        --scene "$scene" \
        --output "$WARP_ROOT/$name/warps" \
        --height "${HEIGHT:-512}" \
        --width "${WIDTH:-640}" \
        --confidence-threshold "${CONFIDENCE_THRESHOLD:-0.2}" \
        --close-kernel "${CLOSE_KERNEL:-3}" \
        --seam-kernel "${SEAM_KERNEL:-3}"
done