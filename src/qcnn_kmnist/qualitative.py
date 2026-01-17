from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from qcnn_kmnist.data.kmnist_datamodule import get_dataloaders
from qcnn_kmnist.utils.io import ensure_dir, load_checkpoint
from qcnn_kmnist.utils.seed import set_seed
from qcnn_kmnist.eval import infer_arch_from_state_dict_keys, pick_device
from qcnn_kmnist.models.baseline_cnn import BaselineCNN
from qcnn_kmnist.models.hybrid_qcnn import HybridQCNN
from qcnn_kmnist.models.matched_classical import MatchedClassicalCNN


def build_model_from_ckpt(ckpt: dict):
    cfg = ckpt.get("config", {})
    state_dict = ckpt["model_state"]
    model_type = cfg.get("model", None)
    inferred = infer_arch_from_state_dict_keys(state_dict)

    if model_type is None:
        model_type = inferred
    if model_type == "baseline":
        model_type = "matched_classical"

    if model_type == "matched_classical":
        model = MatchedClassicalCNN(num_classes=10, n_qubits=int(cfg.get("n_qubits", 6)))
    elif model_type == "hybrid":
        model = HybridQCNN(
            num_classes=10,
            n_qubits=int(cfg.get("n_qubits", 6)),
            n_layers=int(cfg.get("n_layers", 2)),
        )
    elif model_type == "simple_baseline":
        model = BaselineCNN(num_classes=10)
    else:
        # fallback on inferred
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
            raise ValueError(f"Cannot infer model type from checkpoint (model={cfg.get('model')}, inferred={inferred})")

    model.load_state_dict(state_dict)
    return model, model_type


@torch.no_grad()
def collect_batch(model, loader, device, max_items=256):
    model.eval()
    xs, ys, ps, confs = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        conf = torch.max(probs, dim=1).values

        xs.append(x.detach().cpu())
        ys.append(y.detach().cpu())
        ps.append(pred.detach().cpu())
        confs.append(conf.detach().cpu())

        if sum(t.shape[0] for t in xs) >= max_items:
            break

    X = torch.cat(xs, dim=0)[:max_items]
    Y = torch.cat(ys, dim=0)[:max_items]
    P = torch.cat(ps, dim=0)[:max_items]
    C = torch.cat(confs, dim=0)[:max_items]
    return X, Y, P, C


def save_prediction_grid(X, Y, P, C, out_path: Path, title: str, nrows=4, ncols=6):
    fig = plt.figure(figsize=(ncols * 2.2, nrows * 2.2))
    for i in range(nrows * ncols):
        ax = plt.subplot(nrows, ncols, i + 1)
        img = X[i].squeeze(0).numpy()
        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"y={int(Y[i])}, ŷ={int(P[i])}\nconf={float(C[i]):.2f}", fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confidence_hist(C, out_path: Path, title: str):
    fig = plt.figure(figsize=(6, 4))
    plt.hist(C.numpy(), bins=20)
    plt.xlabel("max softmax probability")
    plt.ylabel("count")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_top_errors(X, Y, P, C, out_path: Path, title: str, k=24):
    wrong = (Y != P)
    idx = torch.where(wrong)[0]
    if len(idx) == 0:
        # nothing to save
        return
    # show most confident wrong predictions (interesting mistakes)
    conf_wrong = C[idx]
    sorted_idx = idx[torch.argsort(conf_wrong, descending=True)]
    sorted_idx = sorted_idx[:k]

    ncols = 6
    nrows = int(np.ceil(k / ncols))
    fig = plt.figure(figsize=(ncols * 2.2, nrows * 2.2))
    for j, i in enumerate(sorted_idx):
        ax = plt.subplot(nrows, ncols, j + 1)
        img = X[i].squeeze(0).numpy()
        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"y={int(Y[i])}, ŷ={int(P[i])}\nconf={float(C[i]):.2f}", fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["val", "test"], default="test")
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--max_items", type=int, default=256)
    ap.add_argument("--grid_rows", type=int, default=4)
    ap.add_argument("--grid_cols", type=int, default=6)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt.get("config", {})
    set_seed(int(cfg.get("seed", 42)))

    device = pick_device(args.device if args.device != "auto" else "cpu")

    loaders = get_dataloaders(
        batch_size=128,
        num_workers=2,
        val_fraction=0.1,
        seed=int(cfg.get("seed", 42)),
        data_dir="data_cache",
        download=True,
    )
    loader = loaders.val if args.split == "val" else loaders.test

    model, model_type = build_model_from_ckpt(ckpt)
    model.to(device)

    X, Y, P, C = collect_batch(model, loader, device, max_items=args.max_items)

    run_name = ckpt_path.parent.name
    ckpt_tag = ckpt_path.stem
    out_dir = ensure_dir(Path("outputs/figures") / run_name / f"qualitative_{ckpt_tag}")
    title_base = f"{model_type} | {run_name} | {args.split}"

    save_prediction_grid(
        X, Y, P, C,
        out_dir / f"pred_grid_{args.split}.png",
        title=f"Predictions grid ({title_base})",
        nrows=args.grid_rows,
        ncols=args.grid_cols,
    )
    save_confidence_hist(
        C,
        out_dir / f"confidence_hist_{args.split}.png",
        title=f"Confidence histogram ({title_base})",
    )
    save_top_errors(
        X, Y, P, C,
        out_dir / f"top_errors_{args.split}.png",
        title=f"Most confident errors ({title_base})",
        k=args.grid_rows * args.grid_cols,
    )

    print(f"Saved qualitative figures to: {out_dir}")


if __name__ == "__main__":
    main()
