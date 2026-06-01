"""Evaluate a calibrated checkpoint on the held-out test split.

Reports accuracy, AUROC, Sens@90% specificity, ECE pre/post temperature
scaling. Writes:
  - results/eval_<backbone>.json
  - figures/reliability_<backbone>.pdf (reliability diagram)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from _common import (ROOT, RESULTS, CKPTS, build_backbone, loader, set_seed,
                     compute_metrics, ece, reliability_bins, save_json)

FIG = ROOT / "figures"


def collect(model, dl, device):
    model.eval()
    Z, Y = [], []
    with torch.no_grad():
        for xb, yb in dl:
            Z.append(model(xb.to(device)).cpu())
            Y.append(yb)
    return torch.cat(Z).numpy(), torch.cat(Y).numpy()


def softmax(z, T=1.0):
    z = z / T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def reliability_plot(probs_raw, probs_cal, y, out: Path, n_bins=10):
    """Two-panel reliability diagram with a confidence-histogram inset.

    The histogram tells the reviewer how the test samples are spread across
    the confidence axis; high-AUC classifiers cluster everything in the
    top bin, which a naive reliability curve makes look pathological.
    """
    raw = reliability_bins(probs_raw, y, n_bins, equal_mass=True)
    cal = reliability_bins(probs_cal, y, n_bins, equal_mass=True)
    conf_raw = probs_raw.max(axis=1)
    conf_cal = probs_cal.max(axis=1)

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(4.0, 4.4),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
        sharex=True)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="Perfect")
    if raw:
        x, a, w = zip(*raw)
        ax.plot(x, a, "o-", color="#c0392b", lw=1.0, ms=3.5,
                label=f"Uncalibrated (ECE {ece(probs_raw, y, 15):.3f})")
    if cal:
        x, a, w = zip(*cal)
        ax.plot(x, a, "s-", color="#1f4f8b", lw=1.0, ms=3.0,
                label=f"Temp. scaled (ECE {ece(probs_cal, y, 15):.3f})")
    ax.set_xlim(0.5, 1.005); ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Empirical accuracy")
    ax.grid(alpha=0.2); ax.legend(loc="lower right", fontsize=7, frameon=True)
    # histogram panel
    bins = np.linspace(0.5, 1.0, 26)
    axh.hist(conf_raw, bins=bins, color="#c0392b", alpha=0.45,
             label="Uncalibrated")
    axh.hist(conf_cal, bins=bins, color="#1f4f8b", alpha=0.45,
             label="Temp. scaled")
    axh.set_xlabel("Mean predicted confidence")
    axh.set_ylabel("Count")
    axh.grid(alpha=0.2, axis="y")
    axh.legend(loc="upper left", fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ck["backbone"]
    T = ck.get("temperature", 1.0)
    model = build_backbone(backbone).to(device)
    model.load_state_dict(ck["state_dict"])

    test = pd.read_csv(RESULTS / f"test_split_{backbone}.csv")
    dl_te = loader(test, train=False, batch_size=64)
    Z, Y = collect(model, dl_te, device)

    P_raw = softmax(Z, T=1.0)
    P_cal = softmax(Z, T=T)

    metrics_raw = compute_metrics(P_raw, Y)
    metrics_raw["ece"] = ece(P_raw, Y, n_bins=15)
    metrics_cal = compute_metrics(P_cal, Y)
    metrics_cal["ece"] = ece(P_cal, Y, n_bins=15)
    print(f"raw : {metrics_raw}")
    print(f"cal : {metrics_cal}  T={T:.3f}")

    FIG.mkdir(parents=True, exist_ok=True)
    reliability_plot(P_raw, P_cal, Y, FIG / f"reliability_{backbone}.pdf")

    save_json({"backbone": backbone,
               "test_size": len(test),
               "temperature": T,
               "uncalibrated": metrics_raw,
               "calibrated": metrics_cal,
               "reliability_uncal": reliability_bins(P_raw, Y, 10),
               "reliability_cal":   reliability_bins(P_cal, Y, 10)},
              RESULTS / f"eval_{backbone}.json")
    print(f"figure: {FIG / f'reliability_{backbone}.pdf'}")


if __name__ == "__main__":
    main()
