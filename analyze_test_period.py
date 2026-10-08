#!/usr/bin/env python3
"""Aggregate test_period.csv into the same summaries that feed
results_regression.tex's Table tab:tl_test and Table tab:spatial_summary."""
import pandas as pd

df = pd.read_csv("test_period.csv")

print("=== 1) Mean/Best WI, RMSE across p in {3,6,9,12}, by region and strategy (q=1) ===")
summary = df.groupby(["region", "strategy"]).agg(
    mean_wi=("wi", "mean"), mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"),
    best_wi=("wi", "max"),
).reset_index()
print(summary.to_string(index=False))

print("\n=== 2) Winning strategy per region (by mean WI across p) + which p wins overall ===")
for region in df["region"].unique():
    sub = summary[summary["region"] == region]
    tl = sub[sub["strategy"] == "TL"].iloc[0]
    sc = sub[sub["strategy"] == "Scratch"].iloc[0]
    winner = "TL" if tl["mean_wi"] > sc["mean_wi"] else "Scratch"
    winner_df = df[(df["region"] == region) & (df["strategy"] == winner)]
    best_row = winner_df.loc[winner_df["wi"].idxmax()]
    print(f"{region:14s} winner={winner:8s} mean_WI(TL={tl['mean_wi']:.3f}, Scratch={sc['mean_wi']:.3f})  "
          f"-> best p for winner: p={int(best_row['p'])}, WI={best_row['wi']:.4f}, RMSE={best_row['rmse']:.4f}, MAE={best_row['mae']:.4f}")

print("\n=== 3) Per-region, is the winning strategy consistent across ALL p values? ===")
for region in df["region"].unique():
    sub = df[df["region"] == region]
    piv = sub.pivot(index="p", columns="strategy", values="wi")
    piv["winner"] = piv.apply(lambda r: "TL" if r["TL"] > r["Scratch"] else "Scratch", axis=1)
    print(f"\n{region}:")
    print(piv.to_string())
