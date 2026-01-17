#!/usr/bin/env bash
set -euo pipefail

LATEST_DIR="$(ls -1dt outputs/checkpoints/hybrid_* 2>/dev/null | head -n 1 || true)"

if [[ -z "${LATEST_DIR}" ]]; then
  echo "No hybrid checkpoints found in outputs/checkpoints/hybrid_*"
  echo "Run training first: bash scripts/train_hybrid.sh"
  exit 1
fi

CKPT="${LATEST_DIR}/best.pt"
echo "Evaluating HYBRID: ${CKPT}"
python -m qcnn_kmnist.eval --checkpoint "${CKPT}" --split test --device cpu
