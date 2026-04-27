"""
verify_time_stop.py
===================
P36 — Exit-type breakdown report from the latest walk-forward detail CSV.

Usage:
    python verify_time_stop.py
    python verify_time_stop.py --csv stock_picker_data/results/wf_detail_20240801.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="P36 exit-type breakdown report")
    parser.add_argument("--csv", default=None, help="Path to wf_detail_*.csv")
    args = parser.parse_args()

    results_dir = Path("stock_picker_data/results")

    if args.csv:
        csv_path = Path(args.csv)
    else:
        candidates = sorted(results_dir.glob("wf_detail_*.csv"), reverse=True)
        if not candidates:
            print("[ERROR] No wf_detail_*.csv found in", results_dir)
            sys.exit(1)
        csv_path = candidates[0]

    print(f"\nP36 Exit-Type Breakdown: {csv_path.name}")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[WARN] Detail CSV is empty.")
        sys.exit(0)

    if "exit_type" not in df.columns:
        print("[WARN] 'exit_type' column not present — re-run walk_forward_backtest.py")
        sys.exit(1)

    total = len(df)
    print(f"Total trades: {total}\n")

    summary = (
        df.groupby("exit_type")
        .agg(
            count=("return_5d", "count"),
            pct=("return_5d", lambda x: 100 * len(x) / total),
            median_gross=("return_5d", "median"),
            mean_gross=("return_5d", "mean"),
            win_rate=("positive", "mean"),
        )
        .rename(columns={
            "count": "N",
            "pct": "% of trades",
            "median_gross": "median_gross_ret",
            "mean_gross": "mean_gross_ret",
            "win_rate": "win_rate",
        })
    )
    summary["median_gross_ret"] = (summary["median_gross_ret"] * 100).round(2)
    summary["mean_gross_ret"]   = (summary["mean_gross_ret"]   * 100).round(2)
    summary["win_rate"]         = (summary["win_rate"] * 100).round(1)
    summary["% of trades"]      = summary["% of trades"].round(1)

    print(summary.to_string())

    print("\n--- Time-stop quality check ---")
    ts = df[df["exit_type"] == "TIME_STOP"]
    if ts.empty:
        print("  No TIME_STOP trades.")
    else:
        neg_ts = ts[ts["return_5d"] < 0]
        print(f"  TIME_STOP trades: {len(ts)}")
        print(f"  With negative return: {len(neg_ts)} ({100*len(neg_ts)/len(ts):.1f}%)")
        print(f"  Avg loss on negative TIME_STOPs: {neg_ts['return_5d'].mean()*100:.2f}%")

    print("\n--- Stop quality check ---")
    sh = df[df["exit_type"] == "STOP_HIT"]
    if sh.empty:
        print("  No STOP_HIT trades.")
    else:
        print(f"  STOP_HIT trades: {len(sh)}  ({100*len(sh)/total:.1f}% of all)")
        print(f"  Avg STOP_HIT return: {sh['return_5d'].mean()*100:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
