"""
Train script entrypoint.

Planned usage:
python -m qcnn_kmnist.train --model baseline
python -m qcnn_kmnist.train --model hybrid --n_qubits 6 --n_layers 2
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from qcnn_kmnist.data.kmnist_datamodule import get_dataloaders
from qcnn_kmnist.models.baseline_cnn import BaselineCNN
from qcnn_kmnist.utils.io import ensure_dir, save_checkpoint, save_json
from qcnn_kmnist.utils.metrics import compute_metrics
from qcnn_kmnist.utils.plots import plot_curve
from qcnn_kmnist.utils.seed import set_seed


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    all_true = []
    all_pred = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        losses.append(float(loss.item()))

        pred = torch.argmax(logits, dim=1)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    m = compute_metrics(y_true, y_pred)

    return float(np.mean(losses)), m.accuracy, m.f1_macro


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    running = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running += float(loss.item())
        n_batches += 1

    return running / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["baseline", "hybrid"], required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    # Hybrid params (not used yet, but kept for CLI compatibility)
    parser.add_argument("--n_qubits", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=2)

    args = parser.parse_args()

    if args.model == "hybrid":
        raise SystemExit(
            "Hybrid not implemented yet. Run baseline now. "
            "Next step we will implement PennyLane QuantumLayer + Hybrid model."
        )

    set_seed(args.seed)

    device = pick_device()
    print(f"Device: {device}")

    loaders = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=args.seed,
        data_dir="data_cache",
        download=True,
    )

    model = BaselineCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path("outputs/logs") / f"{args.model}_{run_id}")
    ckpt_dir = ensure_dir(Path("outputs/checkpoints") / f"{args.model}_{run_id}")
    fig_dir = ensure_dir(Path("outputs/figures") / f"{args.model}_{run_id}")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "config": vars(args),
        "device": str(device),
    }

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loaders.train, device, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, loaders.val, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1_macro"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_f1_macro={val_f1:.4f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "config": vars(args),
                },
            )

        # Always save last
        save_checkpoint(
            ckpt_dir / "last.pt",
            {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": vars(args),
            },
        )

        save_json(run_dir / "history.json", history)

    # Final plots
    plot_curve(history["train_loss"], "Train Loss", "loss", fig_dir / "train_loss.png")
    plot_curve(history["val_loss"], "Val Loss", "loss", fig_dir / "val_loss.png")
    plot_curve(history["val_acc"], "Val Accuracy", "acc", fig_dir / "val_acc.png")

    print("\nDone.")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Figures: {fig_dir}")
    print("Best checkpoint:", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
