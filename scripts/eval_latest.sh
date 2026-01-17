#!/usr/bin/env bash
set -euo pipefail

# Evaluate the latest baseline run automatically.
# It expects checkpoints in: outputs/checkpoints/baseline_*/best.pt

LATEST_DIR="$(ls -1dt outputs/checkpoints/baseline_* 2>/dev/null | head -n 1 || true)"

if [[ -z "${LATEST_DIR}" ]]; then
  echo "No baseline checkpoints found in outputs/checkpoints/baseline_*"
  echo "Run training first: bash scripts/train_baseline.sh"
  exit 1
fi

CKPT="${LATEST_DIR}/best.pt"

if [[ ! -f "${CKPT}" ]]; then
  echo "best.pt not found in ${LATEST_DIR}"
  exit 1
fi

echo "Evaluating: ${CKPT}"
python -m qcnn_kmnist.eval --checkpoint "${CKPT}" --split test
