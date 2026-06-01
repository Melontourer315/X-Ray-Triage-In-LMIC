"""Benchmark single-image CPU latency and export the model to ONNX.

Saves checkpoints/<backbone>.onnx and results/latency_<backbone>.json.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch

from _common import (CKPTS, RESULTS, build_backbone, set_seed, save_json,
                     IMG_SIZE)


def measure(model, device, runs: int = 100, warmup: int = 20):
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup): model(x)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter(); model(x); ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    return float(ts.mean()), float(ts.std()), float(np.percentile(ts, 50))


def export_onnx(model, path: Path):
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model.cpu(), x, path, opset_version=17,
                      input_names=["image"], output_names=["logits"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    set_seed()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    backbone = ck["backbone"]
    model = build_backbone(backbone)
    model.load_state_dict(ck["state_dict"])

    cpu_mean, cpu_std, cpu_med = measure(model, torch.device("cpu"))
    print(f"CPU mean={cpu_mean:.1f}ms std={cpu_std:.2f}ms median={cpu_med:.1f}ms")

    onnx_path = CKPTS / f"{backbone}.onnx"
    export_onnx(model, onnx_path)
    print(f"onnx -> {onnx_path}")

    save_json({"backbone": backbone,
               "cpu_latency_ms_mean": cpu_mean,
               "cpu_latency_ms_std":  cpu_std,
               "cpu_latency_ms_median": cpu_med,
               "onnx": str(onnx_path)},
              RESULTS / f"latency_{backbone}.json")


if __name__ == "__main__":
    main()
