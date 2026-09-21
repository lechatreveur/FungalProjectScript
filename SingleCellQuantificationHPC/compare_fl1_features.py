#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-way FL1 comparison on the 11 engineered features: M160 / M161 / M162.

Why this and not a UMAP
-----------------------
The 2026-09-21 null calibration showed silhouette on a UMAP embedding has a
floor of +0.38-0.43 rather than 0, and cannot resolve differences at the effect
sizes seen here (P16). Comparing two continuous blobs that way answers nothing.
The 11 features compared directly need no manifold, no representation learning
and no clustering, and every number is interpretable on its own.

The design
----------
    M160  2026-08-28  `5_1_N1`      400/100 ms, laser 2 @ 5
    M161  2026-09-03  `NeonG_YES_1` 350/120 ms, laser 2 @ 5
    M162  2026-09-09  `NeonG_YES`   350/120 ms, laser 2 @ 5

M161 and M162 share strain, medium AND acquisition settings, differing only in
session and day, so they are session replicates of each other. M160 differs in
strain/medium. That makes the readout:

    M161 ~ M162, both differ from M160   -> strain/medium
    M161 differs from M162               -> session-level factor
    all three agree                      -> neither; look elsewhere

The middle row is the one that tests the stress hypothesis, and it is only
testable because two sessions share a condition.

The stress signature
--------------------
Within M160, laser exposure drives pol1_mid 5.40 -> 1.71, pole distance
1.63 -> 0.31 and Periodicity 0.66 -> 0.28 from FL1 to FL7. If a population is
less stressed, those same quantities should sit HIGHER. So the comparison is
directional, not merely "different": it is checked against that signature.

Statistics
----------
Mann-Whitney U (no normality assumption) with Holm correction across the 11
features x 3 pairs, and Cliff's delta as the effect size, which is the rank
equivalent of "how often does a random cell from A exceed one from B" and is
not distorted by the heavy tails these features have. Effect size is what
matters here: at n in the hundreds almost anything reaches significance.

Run from the repo root with the SSD mounted:
    python3 SingleCellQuantificationHPC/compare_fl1_features.py
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

OUT_ROOT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking")
CTRL = Path("/Volumes/X10 Pro/FungalProject_Outputs/umap_control")

EXPERIMENTS = {
    "M160": dict(folder="2026_08_28_M160", tag="m160", fl1="_FL1_",
                 condition="5_1_N1", exposure="400/100 ms"),
    "M161": dict(folder="2026_09_03_M161", tag="m161", fl1="_FL1_",
                 condition="NeonG_YES_1", exposure="350/120 ms"),
    "M162": dict(folder="2026_09_09_M162", tag="m162", fl1="_FL1_",
                 condition="NeonG_YES", exposure="350/120 ms"),
}

FEATURES = ["pol1_a", "pol1_mid", "pol1_v", "pol2_a", "pol2_mid", "pol2_v",
            "NC_score", "Periodicity", "a1a2", "d", "dd"]

# Quantities that fall under laser stress within M160 (FL1 -> FL7). A less
# stressed population should sit HIGHER on these.
STRESS_DOWN = ["pol1_mid", "d", "Periodicity"]


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b), computed by ranking rather than the O(n*m) pairs."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return np.nan
    joint = np.concatenate([a, b])
    ranks = pd.Series(joint).rank().to_numpy()
    rank_sum_a = ranks[:n_a].sum()
    # U for a over b, converted to the dominance statistic
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    return float(2.0 * u_a / (n_a * n_b) - 1.0)


def magnitude(d: float) -> str:
    """Romano et al. thresholds, the conventional reading of Cliff's delta."""
    ad = abs(d)
    if np.isnan(ad):
        return "n/a"
    if ad < 0.147:
        return "negligible"
    if ad < 0.330:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni, returned in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, float)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()


def load(name: str) -> pd.DataFrame | None:
    spec = EXPERIMENTS[name]
    path = OUT_ROOT / spec["folder"] / "features" / f"umap_features_{spec['tag']}.csv"
    if not path.exists():
        print(f"  {name}: MISSING {path}")
        return None
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["film"].astype(str).str.contains(spec["fl1"], regex=False)].copy()
    df["experiment"] = name
    print(f"  {name}: {len(df)} FL1 datapoints (of {before} total) "
          f"[{spec['condition']}, {spec['exposure']}]")
    return df


def main() -> int:
    print("Loading FL1 features")
    frames = {n: load(n) for n in EXPERIMENTS}
    have = {n: d for n, d in frames.items() if d is not None and len(d)}
    if len(have) < 2:
        print("\nNeed at least two experiments with features built. "
              "Run build_features_m16*.py first.")
        return 1

    missing = [f for f in FEATURES
               if any(f not in d.columns for d in have.values())]
    if missing:
        print(f"\nFeatures absent from at least one table: {missing}")
        return 1

    rows = []
    for fa, fb in combinations(sorted(have), 2):
        a_df, b_df = have[fa], have[fb]
        for feat in FEATURES:
            a = a_df[feat].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            b = b_df[feat].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if len(a) < 5 or len(b) < 5:
                continue
            try:
                p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            except ValueError:
                p = np.nan
            rows.append(dict(pair=f"{fa} vs {fb}", feature=feat,
                             n_a=len(a), n_b=len(b),
                             median_a=float(np.median(a)),
                             median_b=float(np.median(b)),
                             delta=cliffs_delta(a, b), p_raw=p))

    res = pd.DataFrame(rows)
    ok = res["p_raw"].notna()
    res.loc[ok, "p_holm"] = holm(res.loc[ok, "p_raw"].tolist())
    res["magnitude"] = res["delta"].map(magnitude)

    print("\n" + "=" * 78)
    print("Cliff's delta (positive = first experiment higher), Holm-corrected p")
    print("=" * 78)
    for pair, grp in res.groupby("pair", sort=False):
        print(f"\n{pair}")
        print(f"  {'feature':<12} {'median A':>10} {'median B':>10} "
              f"{'delta':>8} {'magnitude':>11} {'p_holm':>10}")
        for _, r in grp.iterrows():
            star = " *" if r["p_holm"] < 0.05 else "  "
            print(f"  {r['feature']:<12} {r['median_a']:>10.3f} {r['median_b']:>10.3f} "
                  f"{r['delta']:>+8.3f} {r['magnitude']:>11} {r['p_holm']:>10.2e}{star}")
        big = grp[grp["magnitude"].isin(["medium", "large"])]
        print(f"  -> {len(big)}/{len(grp)} features at medium or large effect")

    # The design readout: are the replicates closer to each other than to M160?
    print("\n" + "=" * 78)
    print("DESIGN READOUT")
    print("=" * 78)

    def mean_abs_delta(pair: str) -> float:
        g = res[res["pair"] == pair]
        return float(g["delta"].abs().mean()) if len(g) else np.nan

    pairs = {p: mean_abs_delta(p) for p in res["pair"].unique()}
    for p, v in sorted(pairs.items(), key=lambda kv: kv[1]):
        print(f"  mean |delta|  {p:<20} {v:.3f}")

    rep = pairs.get("M161 vs M162")
    cross = [v for k, v in pairs.items() if "M160" in k]
    if rep is not None and cross:
        print()
        if rep < min(cross) * 0.6:
            print("  M161 and M162 agree with each other and both differ from M160")
            print("  -> points to STRAIN/MEDIUM, not session.")
        elif rep > min(cross):
            print("  M161 and M162 differ from each other MORE than from M160")
            print("  -> points to a SESSION-LEVEL factor. This is the row that")
            print("     tests the stress hypothesis directly.")
        else:
            print("  Replicate and cross-condition separations are comparable;")
            print("  no clean attribution. Report the per-feature table instead.")

    # Directional check against M160's own laser-stress signature
    print("\n  Stress-signature check (pol1_mid, d, Periodicity fall under stress):")
    for pair, grp in res.groupby("pair", sort=False):
        sub = grp[grp["feature"].isin(STRESS_DOWN)]
        if not len(sub):
            continue
        signs = np.sign(sub["delta"].to_numpy())
        agree = "consistent" if np.all(signs == signs[0]) and signs[0] != 0 else "mixed"
        direction = ""
        if agree == "consistent":
            first = pair.split(" vs ")[0]
            direction = (f"{first} higher = less stressed"
                         if signs[0] > 0 else f"{first} lower = more stressed")
        print(f"    {pair:<20} {agree:<11} {direction}")

    CTRL.mkdir(parents=True, exist_ok=True)
    csv_path = CTRL / "fl1_feature_comparison.csv"
    res.to_csv(csv_path, index=False)
    print(f"\ntable -> {csv_path}")

    # Per-feature distributions
    names = sorted(have)
    colors = {"M160": "#2563eb", "M161": "#c2410c", "M162": "#16a34a"}
    fig, axes = plt.subplots(3, 4, figsize=(19, 11))
    for ax, feat in zip(axes.ravel(), FEATURES):
        data, labs = [], []
        for n in names:
            v = have[n][feat].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            lo, hi = np.percentile(v, [1, 99]) if len(v) else (0, 1)
            data.append(v[(v >= lo) & (v <= hi)])
            labs.append(f"{n}\nn={len(v)}")
        bp = ax.boxplot(data, labels=labs, showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.6))
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(colors.get(n, "#888"))
            patch.set_alpha(0.45)
        mark = " (falls under stress)" if feat in STRESS_DOWN else ""
        ax.set_title(f"{feat}{mark}", fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(labelsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle("FL1 only, the 11 engineered features — whiskers at 1st/99th "
                 "percentile, outliers hidden", fontsize=13)
    fig.tight_layout()
    png = CTRL / "fl1_feature_comparison.png"
    fig.savefig(png, dpi=140)
    print(f"plot  -> {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
