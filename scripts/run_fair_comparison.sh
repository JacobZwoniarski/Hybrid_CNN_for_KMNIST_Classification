#!/usr/bin/env bash
set -euo pipefail

# Fair comparison runner:
# - trains baseline (matched_classical) and hybrid with identical training params
# - evaluates both on test split
# - writes a summary.json + summary.csv and copies key figures

# -------------------------
# Params (edit if needed)
EPOCHS="${EPOCHS:-15}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-42}"
N_QUBITS="${N_QUBITS:-6}"
N_LAYERS="${N_LAYERS:-2}"
DEVICE="${DEVICE:-cpu}"   # keep cpu for reproducibility + PennyLane compatibility
SPLIT="${SPLIT:-test}"
# -------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/comparisons/${STAMP}"
mkdir -p "${OUT_DIR}"

echo "== Fair comparison =="
echo "epochs=${EPOCHS} lr=${LR} batch=${BATCH_SIZE} wd=${WEIGHT_DECAY} seed=${SEED} n_qubits=${N_QUBITS} n_layers=${N_LAYERS} device=${DEVICE}"
echo "outputs: ${OUT_DIR}"
echo ""

# Ensure package import works
python -c "import qcnn_kmnist; print('qcnn_kmnist import OK')"

# 1) Train baseline (matched)
echo "== Train BASELINE (matched) =="
python -m qcnn_kmnist.train \
  --model baseline \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --n_qubits "${N_QUBITS}" \
  --device "${DEVICE}"

BASELINE_CKPT_DIR="$(ls -1dt outputs/checkpoints/baseline_* | head -n 1)"
BASELINE_CKPT="${BASELINE_CKPT_DIR}/best.pt"
echo "Baseline best checkpoint: ${BASELINE_CKPT}"
echo ""

# 2) Train hybrid
echo "== Train HYBRID =="
python -m qcnn_kmnist.train \
  --model hybrid \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --seed "${SEED}" \
  --n_qubits "${N_QUBITS}" \
  --n_layers "${N_LAYERS}" \
  --device "${DEVICE}"

HYBRID_CKPT_DIR="$(ls -1dt outputs/checkpoints/hybrid_* | head -n 1)"
HYBRID_CKPT="${HYBRID_CKPT_DIR}/best.pt"
echo "Hybrid best checkpoint: ${HYBRID_CKPT}"
echo ""

# 3) Eval both
echo "== Eval BASELINE (matched) on ${SPLIT} =="
python -m qcnn_kmnist.eval --checkpoint "${BASELINE_CKPT}" --split "${SPLIT}" --device "${DEVICE}"

echo "== Eval HYBRID on ${SPLIT} =="
python -m qcnn_kmnist.eval --checkpoint "${HYBRID_CKPT}" --split "${SPLIT}" --device "${DEVICE}"

# Paths to eval outputs
BASELINE_RUN_NAME="$(basename "${BASELINE_CKPT_DIR}")"
HYBRID_RUN_NAME="$(basename "${HYBRID_CKPT_DIR}")"

BASELINE_EVAL_JSON="outputs/predictions/${BASELINE_RUN_NAME}/eval_best/eval_${SPLIT}.json"
HYBRID_EVAL_JSON="outputs/predictions/${HYBRID_RUN_NAME}/eval_best/eval_${SPLIT}.json"

if [[ ! -f "${BASELINE_EVAL_JSON}" ]]; then
  echo "ERROR: Baseline eval json not found: ${BASELINE_EVAL_JSON}"
  exit 1
fi
if [[ ! -f "${HYBRID_EVAL_JSON}" ]]; then
  echo "ERROR: Hybrid eval json not found: ${HYBRID_EVAL_JSON}"
  exit 1
fi

# 4) Extract metrics using python (no jq dependency)
python - <<PY
import json
from pathlib import Path

baseline = json.loads(Path("${BASELINE_EVAL_JSON}").read_text())
hybrid = json.loads(Path("${HYBRID_EVAL_JSON}").read_text())

out_dir = Path("${OUT_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)

summary = {
  "config": {
    "epochs": int("${EPOCHS}"),
    "lr": float("${LR}"),
    "batch_size": int("${BATCH_SIZE}"),
    "weight_decay": float("${WEIGHT_DECAY}"),
    "seed": int("${SEED}"),
    "n_qubits": int("${N_QUBITS}"),
    "n_layers": int("${N_LAYERS}"),
    "device": "${DEVICE}",
    "split": "${SPLIT}",
  },
  "runs": {
    "baseline_matched": {
      "run_name": "${BASELINE_RUN_NAME}",
      "checkpoint": baseline.get("checkpoint"),
      "accuracy": baseline.get("accuracy"),
      "f1_macro": baseline.get("f1_macro"),
    },
    "hybrid": {
      "run_name": "${HYBRID_RUN_NAME}",
      "checkpoint": hybrid.get("checkpoint"),
      "accuracy": hybrid.get("accuracy"),
      "f1_macro": hybrid.get("f1_macro"),
    },
  }
}

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

# CSV
lines = [
  "model,run_name,accuracy,f1_macro",
  f"baseline_matched,{summary['runs']['baseline_matched']['run_name']},{summary['runs']['baseline_matched']['accuracy']},{summary['runs']['baseline_matched']['f1_macro']}",
  f"hybrid,{summary['runs']['hybrid']['run_name']},{summary['runs']['hybrid']['accuracy']},{summary['runs']['hybrid']['f1_macro']}",
]
(out_dir / "summary.csv").write_text("\n".join(lines) + "\n")

print("Wrote:", out_dir / "summary.json")
print("Wrote:", out_dir / "summary.csv")
PY

# 5) Copy key figures to comparison folder (if exist)
BASELINE_FIG_DIR="outputs/figures/${BASELINE_RUN_NAME}/eval_best"
HYBRID_FIG_DIR="outputs/figures/${HYBRID_RUN_NAME}/eval_best"

mkdir -p "${OUT_DIR}/figures/baseline_matched" "${OUT_DIR}/figures/hybrid"

for f in confusion_matrix_${SPLIT}.png confusion_matrix_${SPLIT}_norm.png; do
  if [[ -f "${BASELINE_FIG_DIR}/${f}" ]]; then
    cp "${BASELINE_FIG_DIR}/${f}" "${OUT_DIR}/figures/baseline_matched/${f}"
  fi
  if [[ -f "${HYBRID_FIG_DIR}/${f}" ]]; then
    cp "${HYBRID_FIG_DIR}/${f}" "${OUT_DIR}/figures/hybrid/${f}"
  fi
done

echo ""
echo "== DONE =="
echo "Summary: ${OUT_DIR}/summary.json"
echo "CSV:     ${OUT_DIR}/summary.csv"
echo "Figures: ${OUT_DIR}/figures/"
