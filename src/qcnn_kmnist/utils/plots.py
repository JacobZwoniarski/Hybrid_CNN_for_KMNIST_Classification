"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


def plot_curve(
    values: List[float],
    title: str,
    ylabel: str,
    out_path: str | Path,
    *,
    start_epoch: int = 1,
    force_ylim: Optional[tuple[float, float]] = None,
) -> None:
    """
    Plots a single curve against epoch numbers (start_epoch..start_epoch+N-1),
    with integer x-ticks.

    Args:
        values: list of metric values per epoch (len=N)
        title: plot title
        ylabel: y-axis label
        out_path: where to save the figure
        start_epoch: first epoch number shown on x-axis (default: 1)
        force_ylim: optionally force y-limits, e.g. (0.0, 1.0) for accuracy
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(values)
    if n == 0:
        raise ValueError("values must be a non-empty list")

    x = list(range(start_epoch, start_epoch + n))

    plt.figure()
    plt.plot(x, values)  # no color specified (default)
    plt.title(title)
    
