#!/usr/bin/env python3
"""Extract held-out test-period WI/RMSE/MAE for every region, p in
{3,6,9,12}, q in {1,3,6,9,12}, strategy in {pretrained, scratch}, from
outputs/<region>/analysis/spi3/p<p>_q<q>/<model_type>/metrics/metrics.json
(the authoritative, spi3-namespaced current output; the older
non-namespaced outputs/<region>/inferences/p<p>_q1/.../test_original/ paths
are a stale/pre-namespacing format and are not used).

Writes two files:
  - test_period.csv       q=1 only, in the original column layout (read by
                          analyze_test_period.py and generate_paper_maps.py).
  - test_period_full.csv  every (p, q) combination, with an extra q column.
    The number of test target months is 24 - (p + q) + 1, so long horizons
    rest on very few months (a single one for p=q=12).
"""
import json
import csv
from pathlib import Path

REGIONS = [
    ("Norte", "North"),
    ("Nordeste", "Northeast"),
    ("Centro-Oeste", "Center-West"),
    ("Sudeste", "Southeast"),
    ("Sul", "South"),
]
P_VALUES = [3, 6, 9, 12]
Q_VALUES = [1, 3, 6, 9, 12]
STRATS = [("pretrained", "TL"), ("scratch", "Scratch")]

OUT = Path("outputs")
rows_out = []

for region_pt, region_en in REGIONS:
    for p in P_VALUES:
        for q in Q_VALUES:
            for model_type, strat_label in STRATS:
                path = (OUT / region_pt / "analysis" / "spi3" / f"p{p}_q{q}" / model_type
                        / "metrics" / "metrics.json")
                if not path.exists():
                    print("MISSING", path)
                    continue
                with open(path) as f:
                    d = json.load(f)
                rows_out.append({
                    "region": region_en, "strategy": strat_label, "p": p, "q": q,
                    "wi": d["metrics_wi"], "rmse": d["metrics_rmse"], "mae": d["metrics_mae"],
                    "n_samples": d["n_samples"], "n_valid": d["metrics_n_valid"],
                })

base_fields = ["region", "strategy", "p", "wi", "rmse", "mae", "n_samples", "n_valid"]

with open("test_period.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=base_fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(r for r in rows_out if r["q"] == 1)

with open("test_period_full.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["region", "strategy", "p", "q"] + base_fields[3:])
    w.writeheader()
    w.writerows(rows_out)

n_q1 = sum(r["q"] == 1 for r in rows_out)
print(f"wrote {n_q1} rows to test_period.csv (q=1) and {len(rows_out)} rows to test_period_full.csv")
