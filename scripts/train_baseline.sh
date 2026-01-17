#!/usr/bin/env bash
set -euo pipefail
python -m qcnn_kmnist.train --model baseline --epochs 5
