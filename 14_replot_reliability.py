"""Re-render the reliability diagram with axes that suit a near-saturated
test set.

The original plot used [0.5, 1.0] linear axes with equal-mass bins, but
when the model is at the noise floor (here, ECE 0.008 over 170 test
samples) nearly every sample sits at confidence > 0.95. The diagram
therefore looked like two sparse polygons floating above the diagonal,
which is technically correct but visually misleading.

This version:
  * zooms the x-axis to [0.85, 1.00]
  * plots equal-mass bins as DOTS with marker SIZE proportional to bin count
    (so the eye sees where mass actually is)
  * annotates the bin counts inline
  * adds a tiny inset showing the full [0, 1] picture for context
  * uses a clean caption-ready aspect ratio (4.4 x 3.2)

Output: figures/reliability_resnet18.pdf and .png.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _common import (ROOT, RESULTS, build_backbone, loader, set_seed,
                     reliability_bins, ece)

FIG = ROOT / "figures"


def softmax(z, T=1.0):
    z = z / T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def collect(model, dl, device):
    model.eval(); Z, Y = [], []
    with torch.no_grad():
        for xb, yb in dl:
            Z.append(model(xb.to(device)).cpu()); Y.append(yb)
    return torch.cat(Z).numpy(), torch.cat(Y).numpy()


def plot(probs_raw, probs_cal, y, out: Path):
    """Two-panel reliability figure for a near-saturated test set.

    Left: classical reliability curve, zoomed to [0.85, 1.0] where the data
    actually lives, marker area = bin count.
    Right: bin-count histogram on the same x-axis, so the reader sees the
    confidence mass distribution at a glance.
    """
    n_bins = 10
    raw = reliability_bins(probs_raw, y, n_bins, equal_mass=True)
    cal = reliability_bins(probs_cal, y, n_bins, equal_mass=True)
    ece_raw = ece(probs_raw, y, 15)
    ece_cal = ece(probs_cal, y, 15)
    conf_raw = probs_raw.max(axis=1)
    conf_cal = probs_cal.max(axis=1)

    fig, (ax, axh) = plt.subplots(
        1, 2, figsize=(7.0, 3.0),
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.32})

    # ----- left: reliability curve -----
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, zorder=1)
    def draw(bins, color, marker, label):
        if not bins: return
        xs, ys, ws = zip(*bins)
        sizes = [40 + 4 * w for w in ws]
        ax.plot(xs, ys, "-", color=color, lw=0.9, alpha=0.7, zorder=2)
        ax.scatter(xs, ys, s=sizes, color=color, marker=marker,
                   edgecolor="white", linewidth=0.6, zorder=3, label=label)
    draw(raw, "#c0392b", "o", f"Uncalibrated (ECE {ece_raw:.3f})")
    draw(cal, "#1f4f8b", "s", f"Temp. scaled, $T=0.58$ (ECE {ece_cal:.3f})")
    ax.set_xlim(0.85, 1.005); ax.set_ylim(0.85, 1.005)
    ax.set_xticks([0.85, 0.90, 0.95, 1.00])
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Mean predicted confidence (bin)")
    ax.set_ylabel("Empirical accuracy (bin)")
    ax.grid(alpha=0.25, linewidth=0.3)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.92)
    ax.set_title(f"(a) Reliability, $n={len(y)}$", fontsize=9, loc="left")

    # ----- right: confidence histogram -----
    bins = np.linspace(0.5, 1.0, 26)
    axh.hist(conf_raw, bins=bins, color="#c0392b", alpha=0.55,
             edgecolor="white", linewidth=0.4, label="Uncalibrated")
    axh.hist(conf_cal, bins=bins, color="#1f4f8b", alpha=0.55,
             edgecolor="white", linewidth=0.4, label="Temp. scaled")
    axh.set_xlim(0.5, 1.005)
    axh.set_xlabel("Predicted confidence")
    axh.set_ylabel("Test samples")
    axh.set_title("(b) Confidence histogram", fontsize=9, loc="left")
    axh.grid(alpha=0.25, axis="y", linewidth=0.3)
    axh.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.92)
    # annotate the bulk: most samples sit at >0.95
    frac_above = (conf_cal > 0.95).mean()
    axh.text(0.97, axh.get_ylim()[1] * 0.55,
             f"{frac_above*100:.0f}% of samples\nat conf $>0.95$",
             ha="right", va="top", fontsize=7, color="black",
             bbox=dict(facecolor="white", edgecolor="gray", lw=0.4,
                       boxstyle="round,pad=0.25", alpha=0.92))

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet18_best.pt")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ck["backbone"]
    T = ck.get("temperature", 1.0)
    model = build_backbone(backbone).to(device)
    model.load_state_dict(ck["state_dict"])

    test = pd.read_csv(RESULTS / f"test_split_{backbone}.csv")
    dl = loader(test, train=False, batch_size=64)
    Z, Y = collect(model, dl, device)
    P_raw = softmax(Z, T=1.0)
    P_cal = softmax(Z, T=T)

    out = FIG / f"reliability_{backbone}.pdf"
    plot(P_raw, P_cal, Y, out)
    print(f"saved {out} (and .png)")


if __name__ == "__main__":
    main()
