"""Step 4: fit the three generators, sample, and compare them on fidelity, utility and privacy.

Usage:  python scripts/run_benchmark.py
Writes: results/results.md, results/*.csv, results/*.png
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synthsupply import metrics as M  # noqa: E402
from synthsupply.generators import DomainSimulator, GaussianCopula, IndependentMarginals  # noqa: E402
from synthsupply.prepare import load_reference, split  # noqa: E402

SEED = 42
OUT = ROOT / "results"
MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]
COLORS = {"Real holdout": "#52514e", "Independent": "#2a78d6",
          "Gaussian copula": "#eb6834", "Domain simulator": "#1baf7a"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.axisbelow": True, "font.size": 10})


def main():
    OUT.mkdir(exist_ok=True)
    real = load_reference(ROOT / "data" / "DataCoSupplyChainDataset.csv")
    train, holdout = split(real, seed=SEED)
    print(f"train: {len(train):,}  holdout: {len(holdout):,}")

    generators = {"Independent": IndependentMarginals(seed=SEED),
                  "Gaussian copula": GaussianCopula(seed=SEED),
                  "Domain simulator": DomainSimulator(seed=SEED)}
    synth = {}
    for name, gen in generators.items():
        t = time.time()
        synth[name] = gen.fit(train).sample(len(train))
        print(f"{name:17s} generated in {time.time() - t:.1f}s")

    # The real training set is scored the same way: it is the best any generator could do.
    candidates = {"Real train (reference)": train, **synth}
    rows, pairs = [], {}
    for name, df in candidates.items():
        is_real = name.startswith("Real")
        pairs[name] = M.column_pairs(holdout, df)
        rows.append({
            "generator": name,
            "column_shapes": M.column_shapes(holdout, df).mean(),
            "column_pairs": pairs[name].mean(),
            "rule_validity": M.rule_validity(df)["rule_all"],
            "TSTR_AUC": M.auc_on_holdout(df, holdout, seed=SEED),
            "exact_copy_rate": M.exact_copy_rate(train, holdout if is_real else df),
            "median_DCR": M.median_dcr(train, holdout if is_real else df, seed=SEED),
        })
        print(f"{name:23s} scored")
    summary = pd.DataFrame(rows).set_index("generator")
    summary.to_csv(OUT / "summary.csv")
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / "column_pairs.csv")

    late_by_mode = pd.DataFrame({"Real holdout": holdout.groupby("shipping_mode").late.mean(),
                                 **{k: v.groupby("shipping_mode").late.mean() for k, v in synth.items()}}).loc[MODES]
    late_by_mode.to_csv(OUT / "late_by_mode.csv")

    plot_late_by_mode(late_by_mode)
    plot_signal(holdout, synth["Domain simulator"])
    write_report(summary, pairs, late_by_mode)


def plot_late_by_mode(late_by_mode: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(MODES))
    w = 0.2
    for i, col in enumerate(late_by_mode.columns):
        ax.bar(x + (i - 1.5) * w, late_by_mode[col], width=w * 0.9, color=COLORS[col], label=col)
    ax.set_xticks(x, MODES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of late deliveries")
    ax.set_title("Late rate by shipping mode: only the simulator keeps the real pattern", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(OUT / "late_by_mode.png", dpi=150)
    plt.close(fig)


def plot_signal(holdout: pd.DataFrame, sim: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    for name, df in [("Real holdout", holdout), ("Domain simulator", sim)]:
        m = df.groupby("order_month").late.mean()
        ax.plot(m.index, m.values, marker="o", ms=4, lw=2, color=COLORS[name], label=name)
    ax.set_xticks(range(1, 13))
    ax.set_ylim(0, 1)
    ax.set_xlabel("order month")
    ax.set_ylabel("share of late deliveries")
    ax.set_title("By month: the simulator adds a Nov/Dec peak", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    by_sup = sim.groupby("supplier_id").late.mean().sort_values()
    ax.bar(range(len(by_sup)), by_sup.values, color=COLORS["Domain simulator"], width=0.8)
    ax.axhline(holdout.late.mean(), color=COLORS["Real holdout"], ls="--", lw=1)
    ax.text(-0.4, holdout.late.mean() + 0.03, "real overall rate", ha="left",
            fontsize=8, color=COLORS["Real holdout"])
    ax.set_xticks(range(len(by_sup)), by_sup.index, rotation=90, fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_title("By supplier: hidden reliability creates a learnable signal", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "signal.png", dpi=150)
    plt.close(fig)


def write_report(summary: pd.DataFrame, pairs: pd.DataFrame, late_by_mode: pd.DataFrame):
    weakest = pairs.drop(columns="Real train (reference)")
    weakest = weakest.loc[weakest["Gaussian copula"].nsmallest(5).index]
    md = ["# Benchmark results", "",
          "All metrics compare against the real **holdout** (30% of DataCo, never seen by the generators).",
          "The *Real train* row scores the real training data the same way: the best any generator could do.",
          "", "## Summary", "", summary.round(3).to_markdown(), "",
          "## Late rate by shipping mode", "", late_by_mode.round(3).to_markdown(), "",
          "## The copula's five weakest column pairs", "", weakest.round(3).to_markdown(), ""]
    (OUT / "results.md").write_text("\n".join(md))
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
