#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="."

mkdir -p "${PROJECT_DIR}"/{src/models,src/quantum,scripts,assets,outputs/{checkpoints,logs,figures,metrics},reports/latex}

# ---------- .gitignore ----------
cat > "${PROJECT_DIR}/.gitignore" << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.ipynb_checkpoints/

# Environments
.venv/
venv/
env/
.conda/

# OS / IDE
.DS_Store
.idea/
.vscode/

# Data & outputs (do not commit)
data/
outputs/
reports/build/

# Logs
*.log
EOF

# ---------- requirements.txt ----------
cat > "${PROJECT_DIR}/requirements.txt" << 'EOF'
# Core
numpy
matplotlib
tqdm
pyyaml
scikit-learn

# Deep learning + dataset
torch
torchvision

# Quantum layer
pennylane
pennylane-lightning
EOF

# ---------- README.md ----------
cat > "${PROJECT_DIR}/README.md" << 'EOF'
# Hybrid CNN + Quantum Head on KMNIST (PennyLane + PyTorch)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional) pre-download dataset
python scripts/download_data.py

# Train baseline
python -m src.train --model baseline

# Train hybrid
python -m src.train --model hybrid --n_qubits 6 --q_depth 2

# Evaluate (choose checkpoint)
python -m src.eval --ckpt outputs/checkpoints/latest.pt

# Inference on example input
python -m src.infer --input assets/example_input.npy --output outputs/metrics/example_preds.json
