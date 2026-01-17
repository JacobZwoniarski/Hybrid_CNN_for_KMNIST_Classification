#!/usr/bin/env bash
set -euo pipefail
python -m qcnn_kmnist.train --model hybrid --epochs 5 --n_qubits 6 --n_layers 2
