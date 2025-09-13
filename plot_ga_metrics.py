#!/usr/bin/env python3
"""
Plot GA evolution metrics from evolution_log.csv.

Default: plot best fitness (column 'max') over generations and save to PNG.
"""

from __future__ import annotations

import argparse
import os
import sys
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Plot GA metrics from evolution_log.csv")
    ap.add_argument("--log-csv", type=str, default=os.path.join("ga_results", "evolution_log.csv"))
    ap.add_argument("--out", type=str, default=os.path.join("ga_results", "best_fitness.png"))
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--show", action="store_true", help="Show the figure interactively (requires GUI)")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is required. Please install it: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.log_csv):
        print(f"Log file not found: {args.log_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.log_csv)
    if not {"gen", "max"}.issubset(df.columns):
        print("CSV must contain 'gen' and 'max' columns.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["gen"], df["max"], marker="o", linewidth=1.8)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness (max)")
    ax.set_title("GA Best Fitness over Generations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved plot to {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

