#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
TOPK="${TOPK:-3}"

# Ensure we have example inputs
if [[ ! -d "examples/inputs" ]] || [[ -z "$(ls -1 examples/inputs/*.png 2>/dev/null || true)" ]]; then
  echo "No example inputs found. Generating..."
  bash scripts/generate_examples.sh
fi

echo "Generating baseline vs hybrid comparison panels..."
python -m qcnn_kmnist.infer_compare_plots \
  --inputs_dir examples/inputs \
  --device "${DEVICE}" \
  --topk "${TOPK}"
