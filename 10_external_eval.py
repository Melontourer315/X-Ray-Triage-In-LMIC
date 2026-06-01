"""Out-of-distribution evaluation on TBX11K (China) and AfroCXR (Ethiopia).

The model is trained on Qatar TB-CXR; this script applies it unchanged
(plus the validation-fit temperature) to images sourced from different
populations and reports the same metrics. The drop, if any, is the headline
external-generalisation number reviewers will want to see.

Outputs:
  results/external_<cohort>.json
  figures/external_roc.pdf   (combined ROC: internal vs each external cohort)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from _common import (ROOT, DATA, RESULTS, build_backbone, loader, set_seed,
                     compute_metrics, ece, save_json)

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


def build_tbx11k_index(root: Path) -> pd.DataFrame:
    """TBX11K folder convention: imgs/<class>/<id>.png with class in
    {health, sick, tb, latent_tb, active_tb}. Collapse to binary TB / non-TB.
    """
    rows = []
    for p in root.rglob("*.png"):
        parts = [s.lower() for s in p.parts]
        is_tb = any(s in parts for s in ("tb", "active_tb", "latent_tb"))
        is_neg = any(s in parts for s in ("health", "healthy", "normal"))
        if is_tb:
            rows.append({"path": str(p), "label": 1, "source": "tbx11k"})
        elif is_neg:
            rows.append({"path": str(p), "label": 0, "source": "tbx11k"})
    return pd.DataFrame(rows)


def build_afro_index(root: Path) -> pd.DataFrame:
    """AfroCXR layout varies; we accept either Normal/Tuberculosis subfolders
    or filename-based labels (TB_*/NORMAL_*).
    """
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg"}: continue
        parts = [s.lower() for s in p.parts]
        name = p.stem.lower()
        if any("tuberculosis" in s or s == "tb" for s in parts) or name.startswith("tb"):
            rows.append({"path": str(p), "label": 1, "source": "afro"})
        elif any(s in {"normal", "healthy"} for s in parts) or name.startswith("normal"):
            rows.append({"path": str(p), "label": 0, "source": "afro"})
    return pd.DataFrame(rows)


def build_tbportals_index(root: Path) -> pd.DataFrame:
    """Use the index CSV produced by 11_tbportals_helper.py --parse."""
    idx = RESULTS / "tbportals_index.csv"
    if not idx.exists():
        return pd.DataFrame()
    return pd.read_csv(idx)[["path", "label", "source"]]


COHORTS = {
    "tbx11k":    (DATA / "tbx11k", build_tbx11k_index,
                  "TBX11K (China)"),
    "tbportals": (DATA / "tbportals", build_tbportals_index,
                  "TB Portals (LMIC, incl. South Africa)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cohorts", nargs="+", default=list(COHORTS.keys()))
    ap.add_argument("--max_per_class", type=int, default=500,
                    help="cap each class in external sets for fast eval")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ck["backbone"]; T = ck.get("temperature", 1.0)
    model = build_backbone(backbone).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"loaded {backbone}, T={T:.3f}")

    summary = []
    fpr_tpr = {}                                # for combined ROC
    for cohort in args.cohorts:
        root, builder, pretty = COHORTS[cohort]
        if not root.exists():
            print(f"  skip {cohort}: {root} missing")
            continue
        df = builder(root)
        if df.empty:
            print(f"  skip {cohort}: no images discovered under {root}")
            continue
        # cap each class
        parts = []
        for label, sub in df.groupby("label"):
            parts.append(sub.sample(n=min(args.max_per_class, len(sub)),
                                    random_state=1337))
        df = pd.concat(parts, ignore_index=True)
        print(f"  {cohort}: {len(df)} images "
              f"(TB={(df.label == 1).sum()}, neg={(df.label == 0).sum()})")
        dl = loader(df, train=False, batch_size=64)
        Z, Y = collect(model, dl, device)
        P = softmax(Z, T=T)
        m = compute_metrics(P, Y); m["ece"] = ece(P, Y, n_bins=15)
        m["n"] = len(Y); m["cohort"] = cohort; m["pretty"] = pretty
        summary.append(m)
        # collect ROC
        fpr, tpr, _ = roc_curve(Y, P[:, 1])
        fpr_tpr[cohort] = (fpr, tpr, m["auroc"], pretty)
        save_json(m, RESULTS / f"external_{cohort}.json")
        print(f"  {cohort}: acc={m['accuracy']:.3f} auroc={m['auroc']:.3f} "
              f"sens@90spec={m['sens_at_90_spec']:.3f} ece={m['ece']:.3f}")

    # internal ROC for context
    try:
        ev = json.loads((RESULTS / f"eval_{backbone}.json").read_text())
        # internal ROC isn't saved as curve; redo cheaply by re-reading test split
        test = pd.read_csv(RESULTS / f"test_split_{backbone}.csv")
        dl = loader(test, train=False, batch_size=64)
        Z, Y = collect(model, dl, device); P = softmax(Z, T=T)
        fpr, tpr, _ = roc_curve(Y, P[:, 1])
        fpr_tpr["internal"] = (fpr, tpr, roc_auc_score(Y, P[:, 1]),
                                "Qatar TB-CXR (internal)")
    except Exception as e:
        print(f"  internal ROC skipped: {e}")

    # plot
    if fpr_tpr:
        FIG.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        colors = {"internal": "#1f4f8b", "tbx11k": "#c0392b",
                  "afro": "#27ae60"}
        for k, (fpr, tpr, auc, pretty) in fpr_tpr.items():
            ax.plot(fpr, tpr, lw=1.1, color=colors.get(k, None),
                    label=f"{pretty} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.6)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.2); ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(FIG / "external_roc.pdf", dpi=200)
        plt.close(fig)
        print(f"  saved {FIG / 'external_roc.pdf'}")

    save_json({"backbone": backbone, "rows": summary},
              RESULTS / "external_summary.json")


if __name__ == "__main__":
    main()
