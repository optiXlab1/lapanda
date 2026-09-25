"""Plot the lapanda open-loop loss curves obtained with the forward penalty (1 rho)."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "summary_results"


def load_loss(model: str) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / model / "results" / f"exp2_{model}_open_loop_train_lapanda_fixed32.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return (
        np.asarray([int(row["epoch"]) for row in rows]),
        np.asarray([float(row["loss"]) for row in rows]),
    )


def moving_average(values: np.ndarray, window: int = 15) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    cart_epoch, cart_loss = load_loss("cartpole")
    quad_epoch, quad_loss = load_loss("quadrotor")

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.35), constrained_layout=True)

    panels = [
        (axes[0], cart_epoch, cart_loss, "CartPole", "#1769AA"),
        (axes[1], quad_epoch, quad_loss, "Quadrotor", "#D97706"),
    ]
    for ax, epoch, loss, title, color in panels:
        ax.plot(epoch, loss, color=color, alpha=0.36, linewidth=0.8, label="Per-epoch loss")
        ax.plot(epoch, moving_average(loss), color=color, linewidth=1.8, label="15-epoch moving average")
        ax.scatter([epoch[0], epoch[-1]], [loss[0], loss[-1]], s=22, color=color, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Imitation loss")
        ax.grid(True, color="#D7DEE8", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(epoch[0], loss[0], f"  {loss[0]:.2f}", va="bottom", color="#263238")
        ax.text(epoch[-1], loss[-1], f"{loss[-1]:.2f}  ", ha="right", va="bottom", color="#263238")

    axes[0].legend(frameon=False, loc="upper right", fontsize=8)

    inset = inset_axes(axes[1], width="47%", height="42%", loc="upper right", borderpad=1.0)
    tail = quad_epoch >= 700
    inset.plot(quad_epoch[tail], quad_loss[tail], color="#D97706", linewidth=1.0)
    inset.set_xlim(700, 800)
    inset.set_title("Epochs 700–800", fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(True, color="#D7DEE8", linewidth=0.45, alpha=0.8)
    inset.spines[["top", "right"]].set_visible(False)
    mark_inset(axes[1], inset, loc1=2, loc2=4, fc="none", ec="#7A8793", linewidth=0.7)

    fig.suptitle(r"Open-loop imitation learning with the forward-converged penalty ($1\rho$)", fontsize=12)
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"lapanda_one_rho_loss.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
