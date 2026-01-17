from __future__ import annotations

import argparse
from pathlib import Path

from qcnn_kmnist.utils.io import load_json, ensure_dir
from qcnn_kmnist.utils.plots import plot_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="e.g. outputs/logs/baseline_YYYYMMDD_HHMMSS")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found: {history_path}")

    history = load_json(history_path)

    run_name = run_dir.name
    fig_dir = ensure_dir(Path("outputs/figures") / run_name)

    plot_curve(history["train_loss"], "Train Loss", "Loss", fig_dir / "train_loss.png")
    plot_curve(history["val_loss"], "Val Loss", "Loss", fig_dir / "val_loss.png")
    plot_curve(history["val_acc"], "Val Accuracy", "Accuracy", fig_dir / "val_acc.png", force_ylim=(0.0, 1.0))

    print("Saved figures to:", fig_dir)


if __name__ == "__main__":
    main()
