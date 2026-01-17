#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-cpu}"
SPLIT="${SPLIT:-test}"

BASELINE_DIR="$(ls -1dt outputs/checkpoints/baseline_* 2>/dev/null | head -n 1 || true)"
HYBRID_DIR="$(ls -1dt outputs/checkpoints/hybrid_* 2>/dev/null | head -n 1 || true)"

if [[ -z "${BASELINE_DIR}" || -z "${HYBRID_DIR}" ]]; then
  echo "Need both baseline_* and hybrid_* checkpoints."
  exit 1
fi

echo "Qualitative BASELINE: ${BASELINE_DIR}/best.pt"
python -m qcnn_kmnist.qualitative --checkpoint "${BASELINE_DIR}/best.pt" --split "${SPLIT}" --device "${DEVICE}"

echo "Qualitative HYBRID: ${HYBRID_DIR}/best.pt"
python -m qcnn_kmnist.qualitative --checkpoint "${HYBRID_DIR}/best.pt" --split "${SPLIT}" --device "${DEVICE}"
