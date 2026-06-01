"""Fit a single-parameter temperature T on the validation split (NLL).

Saves T to the checkpoint metadata so 04_evaluate.py can apply it.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from _common import (RESULTS, CKPTS, build_backbone, loader, set_seed, save_json)


def collect_logits(model, dl, device):
    model.eval()
    Z, Y = [], []
    with torch.no_grad():
        for xb, yb in dl:
            Z.append(model(xb.to(device)).cpu())
            Y.append(yb)
    return torch.cat(Z), torch.cat(Y)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Optimise T > 0 to minimise NLL on the held-out (validation) logits."""
    T = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-3), labels)
        loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().clamp_min(1e-3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ck["backbone"]
    model = build_backbone(backbone).to(device)
    model.load_state_dict(ck["state_dict"])

    val = pd.read_csv(RESULTS / f"val_split_{backbone}.csv")
    dl_va = loader(val, train=False, batch_size=64)

    logits, labels = collect_logits(model, dl_va, device)
    T = fit_temperature(logits, labels)
    print(f"fitted temperature T = {T:.4f}")

    # store T inside the checkpoint metadata
    ck["temperature"] = T
    torch.save(ck, args.ckpt)
    save_json({"backbone": backbone, "temperature": T,
               "val_size": len(val)},
              RESULTS / f"calibration_{backbone}.json")
    print(f"saved T into {args.ckpt}")


if __name__ == "__main__":
    main()
