"""Re-render the external-ROC figure to include three cohorts:
internal Qatar test, TBX11K (China), and TB Portals (LMIC, parts 1-3,
paired with Qatar test negatives for an ROC computation).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score

from _common import (ROOT, DATA, RESULTS, build_backbone, loader, set_seed,
                     IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE)
import importlib
g12 = importlib.import_module("12_tbportals_eval")

FIG = ROOT / "figures"


def softmax(z, T=1.0):
    z = z / T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def collect_imgs(model, df, device, T):
    """For Qatar-style PNG paths, returns (probs, labels)."""
    probs, ys = [], []
    buf = []
    def flush():
        nonlocal probs
        if not buf: return
        x = torch.stack([b[0] for b in buf]).to(device)
        with torch.no_grad():
            z = model(x).cpu().numpy()
        probs.append(softmax(z, T=T))
        ys.extend([b[1] for b in buf])
        buf.clear()
    for _, row in df.iterrows():
        try:
            img = Image.open(row["path"]).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            a = (np.asarray(img, dtype=np.float32) / 255.0
                 - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.from_numpy(a.transpose(2, 0, 1)).float()
            buf.append((t, int(row["label"])))
        except Exception:
            continue
        if len(buf) >= 16: flush()
    flush()
    return np.concatenate(probs, axis=0), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/resnet18_best.pt")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ck["backbone"]; T = ck.get("temperature", 1.0)
    model = build_backbone(backbone).to(device); model.eval()
    model.load_state_dict(ck["state_dict"])

    # ---- internal Qatar
    test = pd.read_csv(RESULTS / f"test_split_{backbone}.csv")
    P_q, Y_q = collect_imgs(model, test, device, T)
    fpr_q, tpr_q, _ = roc_curve(Y_q, P_q[:, 1])
    auc_q = roc_auc_score(Y_q, P_q[:, 1])
    print(f"Qatar internal: AUC = {auc_q:.3f}")

    # ---- TBX11K
    from importlib import import_module
    ext = import_module("10_external_eval")
    tbx_dir = DATA / "tbx11k"
    df_tbx = ext.build_tbx11k_index(tbx_dir)
    # subsample to match the eval run (500/500)
    pos = df_tbx[df_tbx.label == 1].sample(n=min(500, (df_tbx.label == 1).sum()),
                                            random_state=1337)
    neg = df_tbx[df_tbx.label == 0].sample(n=min(500, (df_tbx.label == 0).sum()),
                                            random_state=1337)
    df_tbx_eval = pd.concat([pos, neg], ignore_index=True)
    P_t, Y_t = collect_imgs(model, df_tbx_eval, device, T)
    fpr_t, tpr_t, _ = roc_curve(Y_t, P_t[:, 1])
    auc_t = roc_auc_score(Y_t, P_t[:, 1])
    print(f"TBX11K: AUC = {auc_t:.3f}")

    # ---- TB Portals (all positives) paired with Qatar test negatives
    df_p = g12.build_tbportals_index().dropna(subset=["country"])
    paths = df_p["abs_path"].tolist()
    P_p, ok_paths = g12.collect_probs(model, paths, device, T=T, batch_size=16)
    # ROC needs negatives: use Qatar test negatives
    qneg = test[test.label == 0]
    P_qneg, _ = collect_imgs(model, qneg, device, T)
    P_all = np.concatenate([P_p, P_qneg], axis=0)
    Y_all = np.concatenate([np.ones(len(P_p)), np.zeros(len(P_qneg))])
    fpr_p, tpr_p, _ = roc_curve(Y_all, P_all[:, 1])
    auc_p = roc_auc_score(Y_all, P_all[:, 1])
    print(f"TB Portals (LMIC + Qatar negs): AUC = {auc_p:.3f}")

    # ---- TB Portals SSA subset (SA + Nigeria + Senegal) + Qatar negs
    decoded = df_p[df_p["abs_path"].isin(ok_paths)].copy()
    decoded = decoded.set_index("abs_path").loc[ok_paths].reset_index()
    decoded["p_tb"] = P_p[:, 1]
    ssa_mask = decoded["country"].isin(["South Africa", "Nigeria", "Senegal"])
    P_ssa = P_p[ssa_mask.values]
    if len(P_ssa) > 0:
        P_ssa_all = np.concatenate([P_ssa, P_qneg], axis=0)
        Y_ssa = np.concatenate([np.ones(len(P_ssa)), np.zeros(len(P_qneg))])
        fpr_ssa, tpr_ssa, _ = roc_curve(Y_ssa, P_ssa_all[:, 1])
        auc_ssa = roc_auc_score(Y_ssa, P_ssa_all[:, 1])
        print(f"TB Portals SSA subset (n={len(P_ssa)}): AUC = {auc_ssa:.3f}")
    else:
        fpr_ssa = tpr_ssa = None; auc_ssa = float("nan")

    # ---- plot
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.plot(fpr_q, tpr_q, lw=1.4, color="#1f4f8b",
            label=f"Qatar internal, n={len(Y_q)} (AUC {auc_q:.3f})")
    ax.plot(fpr_p, tpr_p, lw=1.2, color="#27ae60",
            label=f"TB Portals LMIC, n={int(Y_all.sum())} (AUC {auc_p:.3f})")
    if fpr_ssa is not None:
        ax.plot(fpr_ssa, tpr_ssa, lw=1.2, color="#e67e22", linestyle="--",
                label=f"TB Portals SSA, n={len(P_ssa)} (AUC {auc_ssa:.3f})")
    ax.plot(fpr_t, tpr_t, lw=1.2, color="#c0392b",
            label=f"TBX11K China, n={len(Y_t)} (AUC {auc_t:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.grid(alpha=0.25, linewidth=0.3)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.92)
    ax.set_title("ROC: internal vs external cohorts", fontsize=9, loc="left")
    fig.tight_layout()
    out = FIG / "external_roc.pdf"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    plt.close(fig)
    print(f"saved {out} (and .png)")


if __name__ == "__main__":
    main()
