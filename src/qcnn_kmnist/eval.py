"""
Eval script entrypoint.

Planned usage:
python -m qcnn_kmnist.eval --checkpoint outputs/checkpoints/...
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from qcnn_kmnist.data.kmnist_datamodule import get_dataloaders
from qcnn_kmnist.models.baseline_cnn import BaselineCNN
from qcnn_kmnist.utils.io import ensure_dir, load_checkpoint, save_json
from qcnn_kmnist.utils.metrics import compute_metrics
from qcnn_kmnist.utils.plots import plot_confusion_matrix
from qcnn_kmnist.utils.seed import set_seed


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def eval_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true = []
    all_pred = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")

    # dataloader params (should match training defaults)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt.get("config", {})
    model_name = cfg.get("model", "baseline")

    # Repro (best-effort)
    set_seed(int(cfg.get("seed", 42)))

    device = pick_device()
    print(f"Device: {device}")

    loaders = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=int(cfg.get("seed", 42)),
        data_dir="data_cache",
        download=True,
    )

    if model_name == "baseline":
        model = BaselineCNN(num_classes=10)
    elif model_name == "hybrid":
        raise SystemExit("Hybrid eval not implemented yet (we will add it after hybrid model).")
    else:
        raise ValueError(f"Unknown model type in checkpoint config: {model_name}")

    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    loader = loaders.val if args.split == "val" else loaders.test
    y_true, y_pred = eval_split(model, loader, device)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Name outputs by run folder: outputs/checkpoints/<run_name>/best.pt -> <run_name>
    run_name = ckpt_path.parent.name
    out_dir = ensure_dir(Path("outputs/predictions") / run_name)
    fig_dir = ensure_dir(Path("outputs/figures") / run_name)

    result = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "accuracy": metrics.accuracy,
        "f1_macro": metrics.f1_macro,
        "confusion_matrix": cm.tolist(),
    }
    save_json(out_dir / f"eval_{args.split}.json", result)

    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(
        cm,
        class_names,
        fig_dir / f"confusion_matrix_{args.split}.png",
        title=f"Confusion Matrix ({args.split})",
        normalize=False,
    )
    plot_confusion_matrix(
        cm,
        class_names,
        fig_dir / f"confusion_matrix_{args.split}_norm.png",
        title=f"Confusion Matrix ({args.split}, normalized)",
        normalize=True,
    )

    print(f"Saved: {out_dir / f'eval_{args.split}.json'}")
    print(f"Saved: {fig_dir / f'confusion_matrix_{args.split}.png'}")
    print(f"Saved: {fig_dir / f'confusion_matrix_{args.split}_norm.png'}")
    print(f"Accuracy={metrics.accuracy:.4f} | F1_macro={metrics.f1_macro:.4f}")


if __name__ == "__main__":
    main()
