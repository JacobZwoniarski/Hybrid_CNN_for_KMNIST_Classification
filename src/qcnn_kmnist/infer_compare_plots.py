from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image

from qcnn_kmnist.infer import (
    KMNIST_CLASS_NAMES,
    build_model_from_ckpt,
    preprocess_image,
    predict,
)
from qcnn_kmnist.utils.io import ensure_dir, load_checkpoint
from qcnn_kmnist.eval import pick_device


def _load_labels_json(labels_path: Path) -> Dict[str, Dict[str, object]]:
    if not labels_path.exists():
        return {}
    data = json.loads(labels_path.read_text(encoding="utf-8"))
    return data.get("labels", {}) if isinstance(data, dict) else {}


def _true_label_from_filename(fname: str) -> Optional[int]:
    # expects ..._y{digit}.png
    m = re.search(r"_y(\d)\.png$", fname)
    if not m:
        return None
    return int(m.group(1))


def _class_label(cid: int) -> str:
    name = KMNIST_CLASS_NAMES[cid] if 0 <= cid < len(KMNIST_CLASS_NAMES) else str(cid)
    return f"{cid} ({name})"


def _auto_latest_checkpoint(prefix: str) -> Path:
    ckpt_root = Path("outputs/checkpoints")
    dirs = sorted(ckpt_root.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No checkpoints found in outputs/checkpoints/{prefix}_*")
    ckpt = dirs[0] / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _plot_topk_barh(ax, topk: List[Tuple[int, float]], title: str) -> None:
    """
    Renders clean horizontal bars for top-k probabilities.
    """
    # Put highest at top (reverse for barh)
    items = list(topk)[::-1]
    labels = [_class_label(cid) for cid, _ in items]
    probs = [float(p) for _, p in items]

    ax.barh(labels, probs)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("probability")
    ax.set_title(title, fontsize=11, pad=6)

    # Add numeric values at bar ends
    for i, p in enumerate(probs):
        ax.text(min(p + 0.02, 0.98), i, f"{p:.3f}", va="center", fontsize=9)

    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.7)


def _make_pretty_panel(
    image_path: Path,
    true_y: Optional[int],
    baseline_pred: int,
    baseline_conf: float,
    baseline_topk: List[Tuple[int, float]],
    hybrid_pred: int,
    hybrid_conf: float,
    hybrid_topk: List[Tuple[int, float]],
    out_path: Path,
) -> None:
    img = Image.open(image_path).convert("L")

    # Title line
    title = image_path.name
    true_str = "unknown"
    if true_y is not None:
        true_str = _class_label(true_y)

    # Correctness marks
    b_ok = (true_y is not None and baseline_pred == true_y)
    h_ok = (true_y is not None and hybrid_pred == true_y)

    b_mark = "✓" if b_ok else "✗"
    h_mark = "✓" if h_ok else "✗"

    b_header = f"Baseline   pred: {_class_label(baseline_pred)} | conf: {baseline_conf:.3f} {b_mark}"
    h_header = f"Hybrid     pred: {_class_label(hybrid_pred)} | conf: {hybrid_conf:.3f} {h_mark}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Layout: image on left (spans 2 rows), bars on right (baseline / hybrid)
    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.0, 1.35],
        height_ratios=[1.0, 1.0],
        wspace=0.25,
        hspace=0.35,
    )

    ax_img = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1])

    # Image panel
    ax_img.imshow(img, cmap="gray")
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title(f"{title}\nTrue: {true_str}", fontsize=12, pad=10)

    # Bars panels
    _plot_topk_barh(ax_b, baseline_topk, b_header)
    _plot_topk_barh(ax_h, hybrid_topk, h_header)

    # Subtle overall title (optional)
    fig.suptitle("Baseline vs Hybrid – Top-3 probabilities", fontsize=13, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs_dir", type=str, default="examples/inputs")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--baseline_ckpt", type=str, default=None)
    ap.add_argument("--hybrid_ckpt", type=str, default=None)
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--mean", type=float, default=0.5)
    ap.add_argument("--std", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=28)
    args = ap.parse_args()

    inputs_dir = Path(args.inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(f"inputs_dir not found: {inputs_dir}")

    labels_map = _load_labels_json(inputs_dir / "labels.json")

    baseline_ckpt = Path(args.baseline_ckpt) if args.baseline_ckpt else _auto_latest_checkpoint("baseline")
    hybrid_ckpt = Path(args.hybrid_ckpt) if args.hybrid_ckpt else _auto_latest_checkpoint("hybrid")

    device = pick_device(args.device if args.device != "auto" else "cpu")

    # Load models
    b_ckpt = load_checkpoint(baseline_ckpt)
    h_ckpt = load_checkpoint(hybrid_ckpt)

    baseline_model, baseline_type = build_model_from_ckpt(b_ckpt)
    hybrid_model, hybrid_type = build_model_from_ckpt(h_ckpt)

    baseline_model.to(device)
    hybrid_model.to(device)

    # Output directory
    if args.out_dir:
        out_dir = ensure_dir(Path(args.out_dir))
    else:
        b_run = baseline_ckpt.parent.name
        h_run = hybrid_ckpt.parent.name
        out_dir = ensure_dir(Path("examples/compare_outputs") / f"{b_run}__vs__{h_run}")

    meta = {
        "inputs_dir": str(inputs_dir),
        "baseline_ckpt": str(baseline_ckpt),
        "hybrid_ckpt": str(hybrid_ckpt),
        "baseline_model_type": baseline_type,
        "hybrid_model_type": hybrid_type,
        "device": str(device),
        "topk": int(args.topk),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    images = sorted([p for p in inputs_dir.glob("*.png") if p.is_file()])
    if not images:
        raise FileNotFoundError(f"No PNG images found in {inputs_dir}")

    results: Dict[str, object] = {"meta": meta, "items": []}

    for img_path in images:
        fname = img_path.name

        # True label
        true_y = None
        if fname in labels_map and isinstance(labels_map[fname], dict) and "y" in labels_map[fname]:
            try:
                true_y = int(labels_map[fname]["y"])
            except Exception:
                true_y = None
        if true_y is None:
            true_y = _true_label_from_filename(fname)

        x = preprocess_image(
            img_path,
            size=args.size,
            mean=args.mean,
            std=args.std,
        )

        b_pred, b_conf, b_top = predict(baseline_model, x, device, topk=args.topk)
        h_pred, h_conf, h_top = predict(hybrid_model, x, device, topk=args.topk)

        out_png = out_dir / f"{img_path.stem}__compare.png"
        _make_pretty_panel(
            img_path,
            true_y,
            b_pred,
            b_conf,
            b_top,
            h_pred,
            h_conf,
            h_top,
            out_png,
        )

        item = {
            "image": fname,
            "true_y": true_y,
            "baseline": {"pred_id": b_pred, "confidence": b_conf, "topk": b_top},
            "hybrid": {"pred_id": h_pred, "confidence": h_conf, "topk": h_top},
            "out_png": str(out_png),
        }
        results["items"].append(item)
        (out_dir / f"{img_path.stem}__compare.json").write_text(json.dumps(item, indent=2), encoding="utf-8")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved nice comparison panels to: {out_dir}")


if __name__ == "__main__":
    main()
