"""Grad-CAM overlays for a sample of correctly-classified TB-positive cases
and a lung-field localisation rate.

Lung field is approximated by Otsu thresholding of the radiograph and the
largest two connected components; the salience peak is "inside the lung field"
if it lies inside that mask. The rate is recorded in
results/gradcam_<backbone>.json and a 4x4 grid of overlays is saved as a PDF.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from _common import (ROOT, RESULTS, build_backbone, set_seed, IMAGENET_MEAN,
                     IMAGENET_STD, save_json)

FIG = ROOT / "figures"
IMG = 224


def preprocess(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG, IMG))
    a = np.asarray(img).astype(np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(a.transpose(2, 0, 1)).float()[None]


def _pick_target_layer(model):
    """Pick the last Conv2d-bearing module. For torchvision ResNets this is
    the last residual block (layer4[-1]); for DenseNet it's the last
    DenseBlock; for VGG it's features[-3] etc. We just walk to the last
    Conv2d in the graph and return its parent so the activation we capture
    has the spatial shape we want.
    """
    import torch.nn as nn
    last_conv = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = (name, m)
    if last_conv is None:
        raise RuntimeError("no Conv2d module found in model")
    return last_conv


def gradcam_simple(model, x, target_class: int):
    """Grad-CAM on the last Conv2d. Returns (cam[H,W], target_layer_name)."""
    feats, grads = {}, {}
    name, mod = _pick_target_layer(model)

    def fwd(_m, _i, o): feats["f"] = o
    def bwd(_m, _gi, go): grads["g"] = go[0]
    h1 = mod.register_forward_hook(fwd)
    h2 = mod.register_full_backward_hook(bwd)

    try:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)
        logits = model(x)
        score = logits[0, target_class]
        model.zero_grad(); score.backward()
        if "f" not in feats or "g" not in grads:
            raise RuntimeError(
                f"hook on {name} did not fire "
                f"(feats={'f' in feats}, grads={'g' in grads})")
        f = feats["f"][0].detach()           # C, H, W
        g = grads["g"][0].detach()           # C, H, W
        w = g.mean(dim=(1, 2))               # C
        cam = (w[:, None, None] * f).sum(0)
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        cam = cv2.resize(cam, (IMG, IMG))
    finally:
        h1.remove(); h2.remove()
    return cam, name


def lung_mask(image_np: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # keep two largest components (left + right lung approximation)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1: return np.zeros_like(m, dtype=bool)
    idx = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:2] + 1
    keep = np.isin(lab, idx)
    return keep


def peak_in_lung(cam: np.ndarray, mask: np.ndarray) -> bool:
    y, x = np.unravel_index(int(cam.argmax()), cam.shape)
    return bool(mask[y, x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=32, help="TB-positive samples to audit")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ck["backbone"]
    model = build_backbone(backbone).to(device)
    model.load_state_dict(ck["state_dict"])
    T = ck.get("temperature", 1.0)

    model.eval()                          # critical for BN; otherwise batch=1
                                          # eval uses noisy batch statistics

    test = pd.read_csv(RESULTS / f"test_split_{backbone}.csv")
    tb = test[test.label == 1].sample(n=min(args.n, len(test[test.label == 1])),
                                      random_state=1337).reset_index(drop=True)

    inside, total, skipped_pred, skipped_err = 0, 0, 0, 0
    panels = []
    for i, row in tb.iterrows():
        try:
            x = preprocess(row.path).to(device)
            with torch.no_grad():
                p = torch.softmax(model(x) / T, dim=1)[0].cpu().numpy()
            pred = int(p.argmax())
            if pred != 1:
                skipped_pred += 1
                continue
            cam, _ = gradcam_simple(model, x, target_class=1)
            raw = np.asarray(
                Image.open(row.path).convert("RGB").resize((IMG, IMG)))
            mask = lung_mask(raw)
            inside += int(peak_in_lung(cam, mask)); total += 1
            if len(panels) < 16:
                panels.append((raw, cam, mask, p[1]))
        except Exception as e:
            skipped_err += 1
            print(f"  skipped {row.path}: {type(e).__name__}: {e}")
    print(f"audit: predicted-TB+={total}, "
          f"missed-by-classifier={skipped_pred}, errors={skipped_err}")

    rate = inside / total if total else float("nan")
    print(f"Grad-CAM peak-in-lung-field: {inside}/{total} = {rate:.3f}")

    # 4x4 overlay grid
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(6.5, 6.5))
    for ax, panel in zip(axes.flat, panels):
        raw, cam, _, prob = panel
        ax.imshow(raw, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.35)
        ax.set_title(f"p={prob:.2f}", fontsize=7); ax.axis("off")
    for ax in axes.flat[len(panels):]: ax.axis("off")
    fig.tight_layout(); fig.savefig(FIG / f"gradcam_{backbone}.pdf", dpi=200)
    plt.close(fig)

    save_json({"backbone": backbone, "n_audited": total,
               "inside_lung_field": inside,
               "lung_localisation_rate": rate},
              RESULTS / f"gradcam_{backbone}.json")


if __name__ == "__main__":
    main()
