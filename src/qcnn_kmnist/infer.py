from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qcnn_kmnist.utils.io import load_checkpoint, ensure_dir
from qcnn_kmnist.utils.seed import set_seed
from qcnn_kmnist.eval import pick_device, infer_arch_from_state_dict_keys
from qcnn_kmnist.models.baseline_cnn import BaselineCNN
from qcnn_kmnist.models.hybrid_qcnn import HybridQCNN
from qcnn_kmnist.models.matched_classical import MatchedClassicalCNN


# KMNIST labels (romanized) – optional, nice for report/demo
KMNIST_CLASS_NAMES = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]


def build_model_from_ckpt(ckpt: dict) -> Tuple[torch.nn.Module, str]:
    cfg = ckpt.get("config", {})
    state_dict = ckpt["model_state"]

    model_type = cfg.get("model", None)
    inferred = infer_arch_from_state_dict_keys(state_dict)

    if model_type is None:
        model_type = inferred

    # In this project baseline == matched classical
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


def preprocess_image(
    image_path: Path,
    *,
    size: int = 28,
    mean: float = 0.5,
    std: float = 0.5,
) -> torch.Tensor:
    """
    Loads a single image and converts it to tensor shape (1,1,H,W).
    Assumes grayscale KMNIST-like input. If you generate examples from torchvision KMNIST,
    this matches well.
    """
    img = Image.open(image_path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), resample=Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]
    # normalize (consistent with typical MNIST/KMNIST pipelines)
    arr = (arr - mean) / std

    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return x


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    topk: int = 3,
) -> Tuple[int, float, List[Tuple[int, float]]]:
    model.eval()
    x = x.to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)  # (10,)

    pred_id = int(torch.argmax(probs).item())
    pred_conf = float(torch.max(probs).item())

    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k=k)
    top = [(int(i.item()), float(v.item())) for i, v in zip(idxs, vals)]
    return pred_id, pred_conf, top


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--out", type=str, default=None, help="Optional output JSON path")
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--topk", type=int, default=3)

    # preprocess params (keep defaults unless you changed training normalization)
    ap.add_argument("--mean", type=float, default=0.5)
    ap.add_argument("--std", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=28)

    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    img_path = Path(args.image)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    ckpt = load_checkpoint(ckpt_path)
    cfg = ckpt.get("config", {})
    set_seed(int(cfg.get("seed", 42)))

    device = pick_device(args.device if args.device != "auto" else "cpu")
    model, model_type = build_model_from_ckpt(ckpt)
    model.to(device)

    x = preprocess_image(img_path, size=args.size, mean=args.mean, std=args.std)
    pred_id, pred_conf, top = predict(model, x, device, topk=args.topk)

    result: Dict[str, object] = {
        "checkpoint": str(ckpt_path),
        "model_type": model_type,
        "image": str(img_path),
        "pred_id": pred_id,
        "pred_name": KMNIST_CLASS_NAMES[pred_id] if 0 <= pred_id < 10 else str(pred_id),
        "confidence": pred_conf,
        "topk": [
            {
                "class_id": cid,
                "class_name": KMNIST_CLASS_NAMES[cid] if 0 <= cid < 10 else str(cid),
                "prob": prob,
            }
            for cid, prob in top
        ],
    }

    out_text = json.dumps(result, indent=2)
    if args.out is None:
        print(out_text)
    else:
        out_path = Path(args.out)
        ensure_dir(out_path.parent)
        out_path.write_text(out_text, encoding="utf-8")
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
