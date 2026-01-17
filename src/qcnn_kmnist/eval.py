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
from qcnn_kmnist.models.hybrid_qcnn import HybridQCNN
from qcnn_kmnist.models.matched_classical import MatchedClassicalCNN
from qcnn_kmnist.utils.io import ensure_dir, load_checkpoint, save_json
from qcnn_kmnist.utils.metrics import compute_metrics
from qcnn_kmnist.utils.plots import plot_confusion_matrix
from qcnn_kmnist.utils.seed import set_seed


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

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


def infer_arch_from_state_dict_keys(state_dict: dict) -> str:
    keys = list(state_dict.keys())

    # MatchedClassicalCNN / HybridQCNN share these blocks:
    if any(k.startswith("pre_quantum.") for k in keys) and any(k.startswith("post_head.") for k in keys):
        if any(k.startswith("quantum.") for k in keys) or any(k.startswith("torch_layer.") for k in keys):
            return "hybrid"
        if any(k.startswith("classical_middle.") for k in keys):
            return "matched_classical"
        # fallback: treat as matched classical if it looks like that pipeline
        return "matched_classical"

    # Old BaselineCNN pattern
    if any(k.startswith("classifier.") for k in keys):
        return "simple_baseline"

    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "mps", "cuda"], default="auto")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt.get("config", {})
    set_seed(int(cfg.get("seed", 42)))

    # Default CPU for safety/repro (and PennyLane compatibility)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cpu"

    device = pick_device(requested_device)
    print(f"Device: {device} (requested={requested_device})")

    loaders = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_fraction=args.val_fraction,
        seed=int(cfg.get("seed", 42)),
        data_dir="data_cache",
        download=True,
    )

    # Decide which model to instantiate:
    model_type = cfg.get("model", None)
    state_dict = ckpt["model_state"]

    # If model_type is missing or ambiguous, infer from keys
    inferred = infer_arch_from_state_dict_keys(state_dict)

    if model_type is None:
        model_type = inferred

    # IMPORTANT: In this project baseline == matched_classical baseline (fair comparison)
    if model_type == "baseline":
        model_type = "matched_classical"

    if model_type == "matched_classical":
        model = MatchedClassicalCNN(
            num_classes=10,
            n_qubits=int(cfg.get("n_qubits", 6)),
        )
    elif model_type == "hybrid":
        model = HybridQCNN(
            num_classes=10,
            n_qubits=int(cfg.get("n_qubits", 6)),
            n_layers=int(cfg.get("n_layers", 2)),
        )
    elif model_type == "simple_baseline":
        model = BaselineCNN(num_classes=10)
    else:
        # last resort: use inferred
        if inferred == "matched_classical":
            model = MatchedClassicalCNN(num_classes=10, n_qubits=int(cfg.get("n_qubits", 6)))
            model_type = "matched_classical"
        elif inferred == "hybrid":
            model = HybridQCNN(
                num_classes=10,
                n_qubits=int(cfg.get("n_qubits", 6)),
                n_layers=int(cfg.get("n_layers", 2)),
            )
            model_type = "hybrid"
        elif inferred == "simple_baseline":
            model = BaselineCNN(num_classes=10)
            model_type = "simple_baseline"
        else:
            raise ValueError(f"Cannot determine model architecture from checkpoint (model={cfg.get('model')}, inferred={inferred})")

    model.load_state_dict(state_dict)
    model.to(device)

    loader = loaders.val if args.split == "val" else loaders.test
    y_true, y_pred = eval_split(model, loader, device)

    metrics = compute_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    run_name = ckpt_path.parent.name
    ckpt_tag = ckpt_path.stem  # best / last

    out_dir = ensure_dir(Path("outputs/predictions") / run_name / f"eval_{ckpt_tag}")
    fig_dir = ensure_dir(Path("outputs/figures") / run_name / f"eval_{ckpt_tag}")

    result = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "model_type": model_type,
        "accuracy": metrics.accuracy,
        "f1_macro": metrics.f1_macro,
        "confusion_matrix": cm.tolist(),
    }
    save_json(out_dir / f"eval_{args.split}.json", result)

    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(
        cm, class_names,
        fig_dir / f"confusion_matrix_{args.split}.png",
        title=f"Confusion Matrix ({args.split})",
        normalize=False,
    )
    plot_confusion_matrix(
        cm, class_names,
        fig_dir / f"confusion_matrix_{args.split}_norm.png",
        title=f"Confusion Matrix ({args.split}, normalized)",
        normalize=True,
    )

    print(f"Accuracy={metrics.accuracy:.4f} | F1_macro={metrics.f1_macro:.4f}")
    print(f"Saved: {out_dir / f'eval_{args.split}.json'}")
    print(f"Saved: {fig_dir / f'confusion_matrix_{args.split}.png'}")
    print(f"Saved: {fig_dir / f'confusion_matrix_{args.split}_norm.png'}")


if __name__ == "__main__":
    main()
