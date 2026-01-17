#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-baseline}"   # baseline or hybrid
DEVICE="${DEVICE:-cpu}"
SPLIT="${SPLIT:-test}"

# Optional: pass checkpoint as first arg
CKPT="${1:-}"

if [[ -z "${CKPT}" ]]; then
  if [[ "${MODEL}" == "hybrid" ]]; then
    DIR="$(ls -1dt outputs/checkpoints/hybrid_* 2>/dev/null | head -n 1 || true)"
  else
    DIR="$(ls -1dt outputs/checkpoints/baseline_* 2>/dev/null | head -n 1 || true)"
  fi

  if [[ -z "${DIR}" ]]; then
    echo "No checkpoints found for MODEL=${MODEL}."
    echo "Train first, e.g.: bash scripts/train_baseline.sh or bash scripts/train_hybrid.sh"
    exit 1
  fi

  CKPT="${DIR}/best.pt"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}"
  exit 1
fi

# Ensure example inputs exist
if [[ ! -d "examples/inputs" ]] || [[ -z "$(ls -1 examples/inputs/*.png 2>/dev/null || true)" ]]; then
  echo "No example inputs found. Generating..."
  bash scripts/generate_examples.sh
fi

RUN_NAME="$(basename "$(dirname "${CKPT}")")"
OUT_DIR="examples/outputs/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "Using checkpoint: ${CKPT}"
echo "Saving outputs to: ${OUT_DIR}"

for img in examples/inputs/*.png; do
  base="$(basename "${img}")"
  out="${OUT_DIR}/${base%.png}.json"
  python -m qcnn_kmnist.infer --checkpoint "${CKPT}" --image "${img}" --out "${out}" --device "${DEVICE}"
done

echo "Done. Example outputs saved under: ${OUT_DIR}"
