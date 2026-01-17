#!/usr/bin/env bash
set -euo pipefail

python -m qcnn_kmnist.generate_examples --out_dir examples/inputs --n 12 --split test
