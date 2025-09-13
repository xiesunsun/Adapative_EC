#!/usr/bin/env python3
"""
Plot population diversity (Hamming fraction) over generations from evolution_log.csv,
and draw a horizontal threshold line (default: historical mean).
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Plot GA population diversity over generations")
    ap.add_argument("--log-csv", type=str, default=os.path.join("ga_results", "evolution_log.csv"))
    ap.add_argument("--out", type=str, default=os.path.join("ga_results", "diversity.png"))
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--no-hline", action="store_true", help="Do not draw horizontal threshold line")
    ap.add_argument("--hline-value", type=float, default=None, help="Custom horizontal line value; overrides mean")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is required. Please install it: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.log_csv):
        print(f"Log file not found: {args.log_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.log_csv)
    if not {"gen", "diversity"}.issubset(df.columns):
        print("CSV must contain 'gen' and 'diversity' columns.", file=sys.stderr)
        sys.exit(1)

    y = df["diversity"].astype(float)
    y_mean = float(np.nanmean(y.values)) if len(y) else float("nan")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["gen"], y, marker="o", linewidth=1.8, label="diversity")

    if not args.no_hline:
        hval = args.hline_value if args.hline_value is not None else y_mean
        if hval == hval:  # not NaN
            ax.axhline(hval, color="red", linestyle="--", alpha=0.7, label=f"threshold={hval:.3f}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Diversity (avg Hamming fraction)")
    ax.set_title("GA Population Diversity over Generations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

