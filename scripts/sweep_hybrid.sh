#!/usr/bin/env bash
set -euo pipefail

# Sweeps a small grid of (n_qubits, n_layers) for HYBRID model only.
# Writes a CSV with metrics + checkpoint paths.

# ---- default params (override via env) ----
EPOCHS="${EPOCHS:-15}"
LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cpu}"
SPLIT="${SPLIT:-test}"

# Grid
GRID="${GRID:-4x1 4x2 6x1 6x2 8x1 8x2 6x3 8x3}"
# ------------------------------------------

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/sweeps/hybrid_${STAMP}"
mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/sweep_results.csv"

echo "model,n_qubits,n_layers,run_name,checkpoint,accuracy,f1_macro" > "${CSV}"

echo "== HYBRID SWEEP =="
echo "grid: ${GRID}"
echo "epochs=${EPOCHS} lr=${LR} batch=${BATCH_SIZE} wd=${WEIGHT_DECAY} seed=${SEED} device=${DEVICE} split=${SPLIT}"
echo "out: ${OUT_DIR}"
echo ""

# Ensure package import works
python -c "import qcnn_kmnist; print('qcnn_kmnist import OK')"

for item in ${GRID}; do
  N_QUBITS="${item%x*}"
  N_LAYERS="${item#*x}"

  echo "----------------------------------------"
  echo "Run: n_qubits=${N_QUBITS} n_layers=${N_LAYERS}"
  echo "----------------------------------------"

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

  RUN_DIR="$(ls -1dt outputs/checkpoints/hybrid_* | head -n 1)"
  CKPT="${RUN_DIR}/best.pt"
  RUN_NAME="$(basename "${RUN_DIR}")"

  python -m qcnn_kmnist.eval --checkpoint "${CKPT}" --split "${SPLIT}" --device "${DEVICE}"

  EVAL_JSON="outputs/predictions/${RUN_NAME}/eval_best/eval_${SPLIT}.json"
  if [[ ! -f "${EVAL_JSON}" ]]; then
    echo "ERROR: eval json not found: ${EVAL_JSON}"
    exit 1
  fi

  # Extract metrics without jq
  python - <<PY
import json
from pathlib import Path
p = Path("${EVAL_JSON}")
d = json.loads(p.read_text())
acc = d.get("accuracy")
f1 = d.get("f1_macro")
print(acc, f1)
PY
  read -r ACC F1 < <(python - <<PY
import json
from pathlib import Path
d = json.loads(Path("${EVAL_JSON}").read_text())
print(d.get("accuracy"), d.get("f1_macro"))
PY
)

  echo "-> acc=${ACC} f1_macro=${F1}"

  echo "hybrid,${N_QUBITS},${N_LAYERS},${RUN_NAME},${CKPT},${ACC},${F1}" >> "${CSV}"

done

echo ""
echo "== DONE =="
echo "CSV: ${CSV}"

# Print top-5 by accuracy
python - <<PY
import csv
from pathlib import Path

csv_path = Path("${CSV}")
rows = []
with csv_path.open() as f:
  r = csv.DictReader(f)
  for row in r:
    row["accuracy"] = float(row["accuracy"])
    row["f1_macro"] = float(row["f1_macro"])
    rows.append(row)

rows.sort(key=lambda x: x["accuracy"], reverse=True)
print("\nTop 5 by accuracy:")
for i, row in enumerate(rows[:5], 1):
  print(f"{i}. n_qubits={row['n_qubits']} n_layers={row['n_layers']} acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f} run={row['run_name']}")
PY
