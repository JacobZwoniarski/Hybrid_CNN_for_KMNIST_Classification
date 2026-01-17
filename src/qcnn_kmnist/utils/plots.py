from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_curve(
    values: List[float],
    title: str,
    ylabel: str,
    out_path: str | Path,
    *,
    start_epoch: int = 1,
    force_ylim: Optional[tuple[float, float]] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(values)
    if n == 0:
        raise ValueError("values must be a non-empty list")

    x = list(range(start_epoch, start_epoch + n))

    plt.figure()
    plt.plot(x, values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.xticks(x)
    plt.xlim(x[0], x[-1])

    if force_ylim is not None:
        plt.ylim(force_ylim[0], force_ylim[1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    out_path: str | Path,
    *,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """
    cm: shape (K, K), integer counts recommended.
    If normalize=True, rows are normalized to sum to 1.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm_int = cm.astype(np.int64)

    if normalize:
        cm_show = cm_int.astype(np.float64)
        row_sums = cm_show.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_show = cm_show / row_sums
        text_values = cm_show  # floats
        fmt = ".2f"
    else:
        cm_show = cm_int.astype(np.float64)  # for imshow scaling
        text_values = cm_int  # ints
        fmt = "d"

    plt.figure(figsize=(8, 7))
    plt.imshow(cm_show, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm_show.max() * 0.6 if cm_show.size else 0.0

    for i in range(text_values.shape[0]):
        for j in range(text_values.shape[1]):
            val = text_values[i, j]
            plt.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                color="white" if cm_show[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
