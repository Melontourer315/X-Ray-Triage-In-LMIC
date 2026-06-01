"""Fine-tune an ImageNet-pretrained backbone with focal loss on Qatar TB-CXR.

Saves the best checkpoint by validation loss to checkpoints/<backbone>_best.pt
and a training log to results/train_<backbone>.json.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from _common import (DATA, CKPTS, RESULTS, set_seed, build_index, make_splits,
                     loader, build_backbone, FocalLoss, save_json, n_params)


def epoch_pass(model, dl, criterion, opt, device, train: bool):
    model.train(train)
    total, count, n_correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in tqdm(dl, leave=False, desc="train" if train else "val"):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(yb)
            count += len(yb)
            n_correct += (logits.argmax(1) == yb).sum().item()
    return total / count, n_correct / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15,
                    help="default 15 is enough for fast-mode subsample; "
                         "use 40 for the full 7k corpus.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--qatar_dir", default=str(DATA / "qatar_tb_cxr"))
    ap.add_argument("--subsample", type=int, default=1000,
                    help="images per class kept BEFORE splitting. "
                         "1000 -> 2000 total, ~5-8 min on GPU. "
                         "Pass 0 (or omit) for the full corpus.")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    df = build_index(Path(args.qatar_dir))
    sub = args.subsample if args.subsample and args.subsample > 0 else None
    splits = make_splits(df, subsample_per_class=sub)
    print(f"split sizes: { {k: len(v) for k, v in splits.items()} }, "
          f"subsample_per_class={sub}")

    dl_tr = loader(splits["train"], train=True,  batch_size=args.batch_size)
    dl_va = loader(splits["val"],   train=False, batch_size=args.batch_size)

    model = build_backbone(args.backbone).to(device)
    print(f"backbone={args.backbone} params={n_params(model):.1f}M")

    crit = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    opt  = Adam(model.parameters(), lr=args.lr)
    sched = ReduceLROnPlateau(opt, factor=0.5, patience=4)

    log, best, stalled = [], float("inf"), 0
    ckpt = CKPTS / f"{args.backbone}_best.pt"
    CKPTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tl, ta = epoch_pass(model, dl_tr, crit, opt, device, train=True)
        vl, va = epoch_pass(model, dl_va, crit, opt, device, train=False)
        sched.step(vl)
        log.append({"epoch": ep, "train_loss": tl, "train_acc": ta,
                    "val_loss": vl, "val_acc": va,
                    "lr": opt.param_groups[0]["lr"]})
        print(f"ep {ep:3d} tl={tl:.4f} ta={ta:.4f} vl={vl:.4f} va={va:.4f}")
        if vl < best - 1e-4:
            best = vl; stalled = 0
            torch.save({"state_dict": model.state_dict(),
                        "backbone": args.backbone,
                        "args": vars(args)}, ckpt)
        else:
            stalled += 1
            if stalled >= args.patience:
                print("early stop"); break

    save_json({"backbone": args.backbone, "log": log,
               "best_val_loss": best,
               "elapsed_sec": time.time() - t0,
               "splits": {k: len(v) for k, v in splits.items()},
               "checkpoint": str(ckpt)},
              RESULTS / f"train_{args.backbone}.json")
    # also save the test split csv so 04_evaluate uses the same one
    splits["test"].to_csv(RESULTS / f"test_split_{args.backbone}.csv", index=False)
    splits["val"].to_csv(RESULTS / f"val_split_{args.backbone}.csv", index=False)
    print(f"done. ckpt={ckpt}")


if __name__ == "__main__":
    main()
