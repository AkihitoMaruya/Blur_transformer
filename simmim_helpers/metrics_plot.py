#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 15:33:34 2025

@author: akihitomaruya
"""
# simmim_helpers/metrics_plot.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metrics_jsonl_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "epoch_metrics.jsonl"


def append_metrics(output_dir: str | Path, row: dict):
    p = metrics_jsonl_path(output_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")


def read_metrics_dedup(output_dir: str | Path) -> list[dict]:
    """
    Reads JSONL, sorts by (epoch, run_ts), keeps last record per epoch.
    """
    p = metrics_jsonl_path(output_dir)
    if not p.is_file():
        return []

    rows = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    rows_sorted = sorted(rows, key=lambda r: (int(r.get("epoch", -1)), float(r.get("run_ts", 0.0))))
    dedup = {}
    for r in rows_sorted:
        e = int(r.get("epoch", -1))
        dedup[e] = r
    return [dedup[e] for e in sorted(dedup.keys()) if e >= 0]


def plot_loss_curves(output_dir: str | Path, *, logger=None) -> Path | None:
    """
    Plot ONLY reconstruction loss:
      - train_rec (blue)
      - val_rec   (red)

    Span + totals are still kept in epoch_metrics.jsonl for later inspection.
    """
    rows = read_metrics_dedup(output_dir)
    if len(rows) == 0:
        if logger:
            logger.warning("[Plot] No epoch metrics found; skipping plot.")
        return None

    epochs = np.array([int(r["epoch"]) for r in rows], dtype=np.int64)

    def _arr(key):
        vals = []
        for r in rows:
            v = r.get(key, None)
            vals.append(np.nan if v is None else float(v))
        return np.array(vals, dtype=np.float64)

    train_rec = _arr("train_rec")
    val_rec   = _arr("val_rec")

    out_path = Path(output_dir) / "loss_curves_rec.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # big + visible
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    plt.figure(figsize=(10, 6))
    lw = 3.5

    if not np.all(np.isnan(train_rec)):
        plt.plot(epochs, train_rec, label="train_rec", linewidth=lw, color="blue")
    if not np.all(np.isnan(val_rec)):
        plt.plot(epochs, val_rec, label="val_rec", linewidth=lw, color="red")

    plt.xlabel("epoch")
    plt.ylabel("reconstruction loss")
    plt.title("SimMIM Reconstruction Loss (train=blue, val=red)")
    plt.tick_params(axis="both", which="both", direction="out", length=7, width=1.8)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    if logger:
        logger.info(f"[Plot] wrote {out_path} (plotted train_rec/val_rec only)")

    return out_path
