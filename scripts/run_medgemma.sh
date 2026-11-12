#!/usr/bin/env bash
# End-to-end run: MedGemma zero-shot generation + LO-VLM-compatible metrics.
#
# Usage:  bash scripts/run_medgemma.sh DATASET_PATH IMAGES_DIR [LIMIT]
set -euo pipefail

DATASET="${1:?usage: run_medgemma.sh DATASET_PATH IMAGES_DIR [LIMIT]}"
IMAGES_DIR="${2:?usage: run_medgemma.sh DATASET_PATH IMAGES_DIR [LIMIT]}"
LIMIT="${3:-}"

mkdir -p outputs

ARGS=(--dataset "$DATASET" --images_dir "$IMAGES_DIR" --output outputs/medgemma4b_test.jsonl)
if [[ -n "$LIMIT" ]]; then ARGS+=(--limit "$LIMIT"); fi

python run_inference.py "${ARGS[@]}"
python run_metrics.py outputs/medgemma4b_test.jsonl --output_json outputs/medgemma4b_metrics.json

echo "Predictions: outputs/medgemma4b_test.jsonl"
echo "Metrics:     outputs/medgemma4b_metrics.json"
