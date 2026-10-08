#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv("grid_full.csv")

print("=== 1) Best-by-WI config per region/strategy (validation) ===")
idx = df.groupby(["region", "strategy"])["wi"].idxmax()
best = df.loc[idx].sort_values(["region", "strategy"])
print(best.to_string(index=False))

print("\n=== 2) q=1 dominance check ===")
for _, row in best.iterrows():
    print(row["region"], row["strategy"], "-> best q =", row["q"], ("OK q=1" if row["q"] == 1 else "*** NOT q=1 ***"))

print("\n=== 3) TL vs Scratch paired comparison (20 configs per region) ===")
for region in df["region"].unique():
    sub = df[df["region"] == region]
    tl = sub[sub["strategy"] == "TL"].set_index(["p", "q"])
    sc = sub[sub["strategy"] == "Scratch"].set_index(["p", "q"])
    merged = tl[["wi", "rmse", "mae"]].join(sc[["wi", "rmse", "mae"]], lsuffix="_tl", rsuffix="_sc")
    merged["d_wi"] = merged["wi_tl"] - merged["wi_sc"]
    merged["d_rmse"] = merged["rmse_tl"] - merged["rmse_sc"]
    n = len(merged)
    win_wi = (merged["d_wi"] > 0).mean()
    win_rmse = (merged["d_rmse"] < 0).mean()  # lower RMSE is a TL "win"
    print(f"{region:14s} n={n:2d}  mean_dWI={merged['d_wi'].mean():+.4f}  win_wi={win_wi:.2f}  "
          f"mean_dRMSE={merged['d_rmse'].mean():+.4f}  win_rmse(lower)={win_rmse:.2f}")

print("\n=== 4) Count of p values selected as best across 10 region-strategy combos ===")
print(best["p"].value_counts().sort_index())

print("\n=== 5) Full WI grid, South, TL ===")
sub = df[(df["region"] == "South") & (df["strategy"] == "TL")]
print(sub.pivot(index="p", columns="q", values="wi").round(4).to_string())

print("\n=== 6) Full WI grid, Center-West, TL ===")
sub = df[(df["region"] == "Center-West") & (df["strategy"] == "TL")]
print(sub.pivot(index="p", columns="q", values="wi").round(4).to_string())
