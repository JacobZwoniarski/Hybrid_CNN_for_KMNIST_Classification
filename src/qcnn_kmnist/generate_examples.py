from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from torchvision.datasets import KMNIST

from qcnn_kmnist.utils.io import ensure_dir

KMNIST_CLASS_NAMES = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="examples/inputs")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", type=str, choices=["train", "test"], default="test")
    ap.add_argument("--data_dir", type=str, default="data_cache")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))

    ds = KMNIST(root=args.data_dir, train=(args.split == "train"), download=True)

    labels = {}
    for i in range(min(args.n, len(ds))):
        img, y = ds[i]  # img is PIL.Image
        fname = f"kmnist_{args.split}_{i:03d}_y{y}.png"
        fpath = out_dir / fname
        img.save(fpath)
        labels[fname] = {"y": int(y), "name": KMNIST_CLASS_NAMES[int(y)]}

    meta = {
        "dataset": "KMNIST",
        "split": args.split,
        "n": len(labels),
        "labels": labels,
    }
    (out_dir / "labels.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {len(labels)} images to: {out_dir}")
    print(f"Saved labels to: {out_dir / 'labels.json'}")


if __name__ == "__main__":
    main()
