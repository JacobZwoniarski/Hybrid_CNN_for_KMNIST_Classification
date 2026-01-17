#!/usr/bin/env bash
set -euo pipefail

python -m qcnn_kmnist.train \
  --model hybrid \
  --epochs 15 \
  --lr 5e-4 \
  --batch_size 128 \
  --weight_decay 0.0 \
  --seed 42 \
  --n_qubits 6 \
  --n_layers 2 \
  --device cpu
