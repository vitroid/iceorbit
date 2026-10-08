"""Figures for the polarization distribution (matplotlib is imported lazily)."""

from __future__ import annotations

from pathlib import Path

from .polarization import MagnitudeDistribution


def plot_magnitude_distribution(dist: MagnitudeDistribution, path: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    width = 0.12
    zero = dist.values == 0

    ax1.bar(dist.values, dist.probability, width=width, color="C0")
    ax1.bar(dist.values[zero], dist.probability[zero], width=width, color="C3", label="|P| = 0")
    ax1.set_ylabel("probability (per configuration)")
    if zero.any():
        p0, n0 = dist.probability[zero][0], dist.n_configs[zero][0]
        ax1.legend(title=f"P(|P|=0) = {p0:.4%}  ({n0:,} / {dist.total:,})", loc="upper right")

    ax2.bar(dist.values, dist.n_classes, width=width, color="C1")
    ax2.bar(dist.values[zero], dist.n_classes[zero], width=width, color="C3")
    ax2.set_ylabel("number of classes")
    ax2.set_xlabel("|P|  (unit bond dipoles)")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
