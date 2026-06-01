"""TB Portals out-of-distribution evaluation.

Reads DICOM images from experiments/data/tbportals/part*/, attaches country
labels from the NIAID manifest + metadata CSV, runs the calibrated ResNet-18
on each image, and reports:

  - Sensitivity at the in-distribution operating threshold, per country and
    in aggregate (TB Portals is TB-only, so this is the only fair clean metric).
  - AUROC + ECE on a synthetic mixed cohort: TB+ from TB Portals paired with
    Normal from the Qatar held-out test split. Documented as such.

Outputs:
  results/tbportals_eval.json
  results/tex/tbportals_rows.tex
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import pydicom
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

from _common import (ROOT, DATA, RESULTS, build_backbone, set_seed,
                     IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, ece, save_json)

PORTAL_ROOT = DATA / "tbportals"
# TB Portals March 2026 release ships a manifest CSV and a patient-level
# metadata CSV. Drop both under data/tbportals/ as:
#   data/tbportals/manifest.csv
#   data/tbportals/metadata.csv
# Override with --meta_csv / --manifest_csv if you keep them elsewhere.
META_CSV     = PORTAL_ROOT / "metadata.csv"
MANIFEST_CSV = PORTAL_ROOT / "manifest.csv"


def dicom_to_tensor(path: Path) -> torch.Tensor | None:
    """DICOM -> 224x224 RGB tensor, ImageNet-normalised. None if the file
    cannot be decoded (e.g. compressed transfer syntax without GDCM)."""
    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr.mean(axis=-1) if arr.shape[-1] == 3 else arr[arr.shape[0] // 2]
    # Apply DICOM windowing if MONOCHROME1 (white = low intensity)
    photo = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photo == "MONOCHROME1":
        arr = arr.max() - arr
    # Robust normalisation: percentile clip avoids saturated borders dominating
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(a.transpose(2, 0, 1)).float()


def softmax_T(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def build_tbportals_index() -> pd.DataFrame:
    """Walk part01/ and part02/, join to manifest+meta, return a frame with
    columns: short_path (absolute), country, cxr_outlier, diagnosis_code."""
    meta = pd.read_csv(META_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)
    rows = []
    for part_dir in sorted(PORTAL_ROOT.glob("part*")):
        idx_csv = part_dir / "_index.csv"
        if not idx_csv.exists():
            print(f"  skip {part_dir.name}: no _index.csv"); continue
        idx = pd.read_csv(idx_csv)
        idx["abs_path"] = idx["short_name"].map(lambda s: str(part_dir / s))
        rows.append(idx)
    if not rows:
        raise SystemExit("No part*/_index.csv found")
    idx = pd.concat(rows, ignore_index=True)
    print(f"local DICOM files: {len(idx)}")
    df = idx.merge(manifest, left_on="orig_path", right_on="file", how="left")
    df = df.merge(meta, left_on="file", right_on="series_instance_content_url",
                  how="left")
    print(f"  with country label: {df['country'].notna().sum()}")
    return df


def collect_probs(model, paths: list[Path], device, T: float, batch_size: int = 16):
    """Iterate over file paths, batch them through the model, return probs
    and the subset of paths that decoded successfully."""
    probs, ok_paths = [], []
    buf, buf_paths = [], []

    def flush():
        if not buf: return
        x = torch.stack(buf).to(device)
        with torch.no_grad():
            z = model(x).cpu().numpy()
        p = softmax_T(z, T=T)
        probs.append(p)
        ok_paths.extend(buf_paths)
        buf.clear(); buf_paths.clear()

    t0 = time.time()
    for i, p in enumerate(paths):
        t = dicom_to_tensor(Path(p))
        if t is None: continue
        buf.append(t); buf_paths.append(p)
        if len(buf) >= batch_size: flush()
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(paths)}  ok={len(ok_paths)}  {time.time()-t0:.0f}s")
    flush()
    P = np.concatenate(probs, axis=0) if probs else np.zeros((0, 2))
    return P, ok_paths


def sens_at_threshold(p_tb: np.ndarray, threshold: float) -> float:
    """Sensitivity for a TB+-only cohort: fraction with p(TB) >= threshold."""
    return float((p_tb >= threshold).mean()) if len(p_tb) else float("nan")


def get_id_threshold(model, device, T: float, target_spec: float = 0.90) -> float:
    """Refit the threshold corresponding to Sens@90 spec on the Qatar val split
    so we apply a *fair* in-distribution operating point to TB Portals."""
    val = pd.read_csv(RESULTS / "val_split_resnet18.csv")
    from PIL import Image as PILImage
    probs, ys = [], []
    buf = []
    def flush():
        nonlocal probs
        if not buf: return
        x = torch.stack([b[0] for b in buf]).to(device)
        with torch.no_grad():
            z = model(x).cpu().numpy()
        probs.append(softmax_T(z, T=T))
        ys.extend([b[1] for b in buf])
        buf.clear()
    for _, row in val.iterrows():
        try:
            img = PILImage.open(row["path"]).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
            a = (np.asarray(img, dtype=np.float32) / 255.0
                 - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.from_numpy(a.transpose(2, 0, 1)).float()
            buf.append((t, int(row["label"])))
        except Exception:
            continue
        if len(buf) >= 16: flush()
    flush()
    P = np.concatenate(probs, axis=0)
    Y = np.array(ys)
    fpr, tpr, thr = roc_curve(Y, P[:, 1])
    spec = 1 - fpr
    # smallest threshold whose specificity >= target_spec
    ok = np.where(spec >= target_spec)[0]
    if len(ok) == 0: return 0.5
    j = ok[tpr[ok].argmax()]
    return float(thr[j])


def synthetic_ood_metrics(p_tb_portals: np.ndarray, model, device, T: float,
                          n_neg: int = 500) -> dict:
    """Build a synthetic OOD set: all TB+ from TB Portals + n_neg Normal from
    the Qatar test split. Returns AUROC and ECE. The cohort mix is documented
    in the reporting layer."""
    test = pd.read_csv(RESULTS / "test_split_resnet18.csv")
    negs = test[test["label"] == 0].sample(n=min(n_neg, len(test[test["label"] == 0])),
                                           random_state=1337)
    from PIL import Image as PILImage
    probs_neg = []; buf = []
    def flush():
        nonlocal probs_neg
        if not buf: return
        x = torch.stack(buf).to(device)
        with torch.no_grad():
            z = model(x).cpu().numpy()
        probs_neg.append(softmax_T(z, T=T))
        buf.clear()
    for _, row in negs.iterrows():
        try:
            img = PILImage.open(row["path"]).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE), PILImage.BILINEAR)
            a = (np.asarray(img, dtype=np.float32) / 255.0
                 - IMAGENET_MEAN) / IMAGENET_STD
            buf.append(torch.from_numpy(a.transpose(2, 0, 1)).float())
        except Exception:
            continue
        if len(buf) >= 16: flush()
    flush()
    Pn = np.concatenate(probs_neg, axis=0)
    P = np.concatenate([p_tb_portals, Pn], axis=0)
    Y = np.concatenate([np.ones(len(p_tb_portals)), np.zeros(len(Pn))])
    auc = float(roc_auc_score(Y, P[:, 1]))
    # ECE on the *predicted-class* confidence as elsewhere
    eval_ece = ece(P, Y.astype(int), n_bins=15)
    pred = (P[:, 1] >= 0.5).astype(int)
    acc = float((pred == Y).mean())
    fpr, tpr, _ = roc_curve(Y, P[:, 1])
    spec = 1 - fpr
    sens90 = float(tpr[spec >= 0.90].max()) if (spec >= 0.90).any() else float("nan")
    return {"n_pos": len(p_tb_portals), "n_neg": len(Pn),
            "accuracy": acc, "auroc": auc, "sens_at_90_spec": sens90,
            "ece": float(eval_ece)}


def main():
    global META_CSV, MANIFEST_CSV
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total images for a quick sanity run")
    ap.add_argument("--meta_csv",     default=str(META_CSV),
                    help="path to TB_Portals_CXRs_March_2026.csv (patient meta)")
    ap.add_argument("--manifest_csv", default=str(MANIFEST_CSV),
                    help="path to TB_Portals_CXRs_March_2026_manifest.csv")
    args = ap.parse_args()
    META_CSV     = Path(args.meta_csv)
    MANIFEST_CSV = Path(args.manifest_csv)

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_backbone(ck["backbone"]).to(device); model.eval()
    model.load_state_dict(ck["state_dict"])
    T = ck.get("temperature", 1.0)
    print(f"backbone={ck['backbone']}  T={T:.3f}  device={device}")

    df = build_tbportals_index()
    df = df.dropna(subset=["country"])
    if args.limit: df = df.head(args.limit)
    paths = df["abs_path"].tolist()
    print(f"running on {len(paths)} TB Portals images")

    # In-distribution threshold for Sens@90 spec
    tau_id = get_id_threshold(model, device, T, target_spec=0.90)
    print(f"in-distribution threshold tau* (Sens@90spec on Qatar val) = {tau_id:.4f}")

    P, ok_paths = collect_probs(model, paths, device, T=T)
    p_tb = P[:, 1]
    print(f"decoded {len(ok_paths)}/{len(paths)} DICOMs")

    # Attach country to decoded subset
    decoded = df[df["abs_path"].isin(ok_paths)].copy()
    decoded = decoded.set_index("abs_path").loc[ok_paths].reset_index()
    decoded["p_tb"] = p_tb
    decoded["pred_tb"] = (decoded["p_tb"] >= tau_id).astype(int)

    # Per-country sensitivity at tau_id
    by_country = []
    for c, g in decoded.groupby("country"):
        s = sens_at_threshold(g["p_tb"].values, tau_id)
        mean_p = float(g["p_tb"].mean())
        by_country.append({"country": c, "n": int(len(g)),
                           "sens_at_tau_id": round(s, 4),
                           "mean_p_tb": round(mean_p, 4)})
    by_country.sort(key=lambda r: -r["n"])

    # Aggregate rows
    all_sens = sens_at_threshold(decoded["p_tb"].values, tau_id)
    ssa = decoded[decoded["country"].isin(["South Africa", "Nigeria", "Senegal"])]
    ssa_sens = sens_at_threshold(ssa["p_tb"].values, tau_id)

    # Synthetic OOD metrics
    syn = synthetic_ood_metrics(P, model, device, T)

    out = {
        "backbone": ck["backbone"],
        "temperature": T,
        "tau_id_at_90spec": tau_id,
        "n_total_decoded": len(decoded),
        "n_total_attempted": len(paths),
        "sens_overall": all_sens,
        "sens_ssa_subset": {"n": int(len(ssa)), "sens": ssa_sens},
        "by_country": by_country,
        "synthetic_ood": syn,
    }
    save_json(out, RESULTS / "tbportals_eval.json")

    # LaTeX fragment
    lines = []
    lines.append(f"All TB Portals (LMIC, parts 1--2) & {len(decoded)} & "
                 f"{all_sens:.3f} & --- & --- & --- \\\\")
    lines.append(f"\\quad Sub-Saharan Africa subset & {len(ssa)} & "
                 f"{ssa_sens:.3f} & --- & --- & --- \\\\")
    lines.append(f"TB Portals TB+ {chr(43)} Qatar normals (synthetic) & "
                 f"{syn['n_pos']}+{syn['n_neg']} & "
                 f"{syn['accuracy']:.3f} & {syn['auroc']:.3f} & "
                 f"{syn['sens_at_90_spec']:.3f} & {syn['ece']:.3f} \\\\")
    (RESULTS / "tex" / "tbportals_rows.tex").write_text("\n".join(lines))

    print("\n=== Summary ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
