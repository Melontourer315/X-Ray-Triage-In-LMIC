"""Download Qatar TB-CXR (primary) and TBX11K (external test).

Both datasets are pulled from Kaggle. Requires kaggle.json at ~/.kaggle/.
Run 00_setup.py first to verify credentials.
"""
from __future__ import annotations
import os, sys, zipfile, shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DATASETS = [
    # (kaggle slug, target subfolder, description)
    ("tawsifurrahman/tuberculosis-tb-chest-xray-dataset", "qatar_tb_cxr",
     "PRIMARY: TB and Normal classes, ~4200 PNGs at 512x512 (Rahman 2020 v1)"),
    ("usmanshams/tbx-11", "tbx11k",
     "EXTERNAL: TBX11K — 11,200 CXRs (Liu 2020), multi-class TB labels, "
     "Chinese clinical cohort. Used for out-of-distribution evaluation."),
    ("raddar/chest-xrays-tuberculosis-from-india", "india_tb",
     "EXTERNAL (LMIC): Indian TB-CXR cohort (small, ~150 images, TB only). "
     "Used as a second OOD cohort. Skip silently if 404."),
]

# Note: no openly-downloadable African TB CXR cohort exists as of 2026.
# Every African release (TB Portals South Africa, CHARM, CAPTURE) is DUA-gated.
# We use TBX11K (China) + Indian-TB-CXR as LMIC external validation and document
# the African data gap in the discussion.

def kaggle_download(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        print(f"  already populated: {dest}, skipping. rm -rf to re-download.")
        return
    # Use the kaggle Python API directly — does not depend on the kaggle CLI
    # being on PATH.
    print(f"  kaggle.api.dataset_download_files({slug}, path={dest}, unzip=True)")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)

def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for slug, sub, descr in DATASETS:
        target = DATA / sub
        print(f"\n>>> {slug}\n    {descr}\n    -> {target}")
        try:
            kaggle_download(slug, target)
        except Exception as e:
            print(f"  WARN: download failed ({e}); continuing.")

    print("\nDone. Top-level contents of data/:")
    for p in sorted(DATA.iterdir()):
        if p.is_dir():
            print(f"  {p.name}/")

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:
        print("kaggle CLI not on PATH. pip install kaggle, then re-run.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"kaggle download failed: {e}")
        print("Check that ~/.kaggle/kaggle.json exists and is chmod 600.")
        sys.exit(1)
