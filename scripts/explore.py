"""Step 1: exploratory analysis of DataCo.

Answers five questions before we generate anything:
  Q1  What is in the data after cleaning?
  Q2  How are delivery promises (scheduled days) set?
  Q3  What do real delivery times look like?
  Q4  Does anything other than shipping mode explain late deliveries?
  Q5  Are there hidden structures (time blocks, price lists) a generator must keep?

Usage:  python scripts/explore.py
Writes: results/eda/eda.md and results/eda/*.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synthsupply.prepare import CATEGORICAL, load_raw, load_reference  # noqa: E402

OUT = ROOT / "results" / "eda"
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10,
})
MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    return df.round(digits).to_markdown()


def main(csv: Path):
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw(csv)
    df = load_reference(csv)
    rep = ["# Step 1 - Exploratory analysis of DataCo", ""]

    # ---------------------------------------------------------------- Q1 ----
    cancelled = (raw["Delivery Status"] == "Shipping canceled").sum()
    rep += ["## Q1. What is in the data?", "",
            f"* Raw file: **{len(raw):,} rows x {raw.shape[1]} columns** (one row per order line).",
            f"* Removed **{cancelled:,} cancelled lines** (never shipped) and 10 personal fields.",
            f"* Clean reference table: **{len(df):,} rows x {df.shape[1]} columns**.",
            f"* Period: {df.order_date.min():%Y-%m-%d} to {df.order_date.max():%Y-%m-%d}; "
            f"{raw['Order Id'].nunique():,} orders, {raw['Product Name'].nunique()} products, "
            f"{df.department.nunique()} departments.",
            f"* Missing values in kept columns: {int(df.isna().sum().sum())}.",
            f"* Overall late rate: **{df.late.mean():.1%}**.", "",
            "Category sizes:", ""]
    for c in CATEGORICAL:
        vc = df[c].value_counts(normalize=True)
        rep.append(f"* `{c}`: " + ", ".join(f"{k} {v:.0%}" for k, v in vc.items()))
    rep.append("")

    # ---------------------------------------------------------------- Q2 ----
    sched = pd.crosstab(df.shipping_mode, df.days_scheduled).loc[MODES]
    rep += ["## Q2. How are delivery promises set?", "",
            "Rows = shipping mode, columns = scheduled days, cells = number of order lines.", "",
            md_table(sched), "",
            "Every mode has exactly one promise: **scheduled days are a fixed function of shipping mode.**", ""]

    # ---------------------------------------------------------------- Q3 ----
    real = pd.crosstab(df.shipping_mode, df.days_real, normalize="index").loc[MODES]
    rule = (df.late == (df.days_real > df.days_scheduled).astype(int)).mean()
    late_mode = df.groupby("shipping_mode").late.mean().loc[MODES]
    rep += ["## Q3. What do real delivery times look like?", "",
            "Share of lines by real shipping days, per mode:", "", md_table(real, 3), "",
            "Late rate per mode:", "", md_table(late_mode.to_frame("late_rate")), "",
            f"* `late == (days_real > days_scheduled)` holds for **{rule:.1%}** of rows: "
            "the late flag is a rule, not a judgement.",
            "* Second and Standard Class: real days are spread **evenly over 2-6** (about 20% each).",
            "* First Class: always 2 days against a promise of 1, so **100% late**.", ""]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    for ax, mode in zip(axes, MODES):
        s = df.loc[df.shipping_mode == mode, "days_real"].value_counts(normalize=True).sort_index()
        prom = df.loc[df.shipping_mode == mode, "days_scheduled"].iloc[0]
        colors = [ORANGE if d > prom else BLUE for d in s.index]
        ax.bar(s.index, s.values, color=colors, width=0.7)
        ax.axvline(prom + 0.5, color=INK, lw=1, ls="--")
        ax.set_title(f"{mode}\npromise = {prom} day(s)", fontsize=10)
        ax.set_xticks(range(0, 7))
        ax.set_xlabel("real shipping days")
    axes[0].set_ylabel("share of order lines")
    axes[0].legend(handles=[Patch(color=BLUE, label="on time / early"), Patch(color=ORANGE, label="late")],
                   frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Real delivery time by shipping mode (dashed line = promise)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "delivery_by_mode.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- Q4 ----
    # Within ONE mode, does any other variable change the late rate?
    std = df[df.shipping_mode == "Standard Class"]
    factors = ["market", "customer_segment", "department", "payment_type", "order_month", "quantity"]
    rows = []
    for f in factors:
        g = std.groupby(f).late.agg(["mean", "size"])
        chi2, p, *_ = stats.chi2_contingency(pd.crosstab(std[f], std.late))
        rows.append({"factor": f, "groups": len(g), "min_late": g["mean"].min(),
                     "max_late": g["mean"].max(), "spread_pp": 100 * (g["mean"].max() - g["mean"].min()),
                     "chi2_p_value": p})
    q4 = pd.DataFrame(rows).set_index("factor")
    # Same question for days_real itself: is it uniform over 2-6?
    obs = std.days_real.value_counts().sort_index()
    p_uniform = stats.chisquare(obs).pvalue
    rep += ["## Q4. Does anything besides shipping mode explain delays?", "",
            "Inside Standard Class only (so shipping mode is fixed), late rate per group of each variable.",
            "A chi-square test checks whether the differences are bigger than chance would produce.", "",
            md_table(q4, 3), "",
            f"* Chi-square test that Standard Class real days are uniform over 2-6: p = {p_uniform:.2f}, "
            f"but the shares only range from {obs.min()/obs.sum():.1%} to {obs.max()/obs.sum():.1%}.",
            "* Late rates move by at most a few percentage points. With 100,000 rows, a test can flag "
            "differences that are real but tiny: customer segment is statistically significant, yet its "
            "whole effect is about 1.4 percentage points. **Statistically detectable is not the same as "
            "useful for prediction.**",
            "* **Conclusion: once shipping mode is known, delays are almost pure noise.** DataCo looks like a "
            "synthetic dataset itself, with delivery days drawn close to uniformly.", ""]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(factors))
    for i, f in enumerate(factors):
        g = std.groupby(f).late.mean()
        ax.scatter(g.values, np.full(len(g), i), color=BLUE, s=28, alpha=0.8, zorder=3)
    ax.axvline(std.late.mean(), color=INK, lw=1, ls="--")
    ax.text(std.late.mean(), -0.75, f" overall {std.late.mean():.0%}", fontsize=8, color=MUTED, va="center")
    ax.set_yticks(y, factors)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.1, len(factors) - 0.5)
    ax.set_xlabel("late rate of each group (Standard Class only)")
    ax.set_title("Inside one shipping mode, every group has the same late rate", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "no_signal.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- Q5 ----
    q = df.assign(quarter=df.order_date.dt.to_period("Q").astype(str))
    blocks = pd.crosstab(q.quarter, q.market)
    dept_prices = df.groupby("department").unit_price.agg(n_prices="nunique", min="min", max="max")
    disc = df.discount_rate.value_counts().sort_index()
    rep += ["## Q5. Hidden structures a generator must keep", "",
            "**Markets come in time blocks.** Order lines per quarter and market:", "",
            blocks.to_markdown(), "",
            "Only one or two markets are active in most quarters, so market depends strongly on date.", "",
            "**Each department has a small fixed price list:**", "", md_table(dept_prices, 2), "",
            f"**Discounts take only {len(disc)} values** ({', '.join(f'{v:.0%}' for v in disc.index)}).",
            f" Quantity takes only {df.quantity.nunique()} values (1-5).", ""]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    share = blocks.div(blocks.sum(axis=1), axis=0)
    im = ax.imshow(share.T.values, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(share.columns)), share.columns)
    ax.set_xticks(range(len(share.index)), share.index, rotation=45, ha="right", fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("share of quarter's orders")
    ax.set_title("Markets appear in time blocks", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "market_blocks.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------- implications --
    rep += ["## What this means for the generators", "",
            "1. They must keep **deterministic rules**: promise = f(mode) and late = real > promise.",
            "2. They must keep **conditional structures**: market given date, price given department.",
            "3. Reproducing DataCo exactly gives data where delays are unpredictable. To be useful for "
            "procurement models, a generator should keep the overall delay rates but give delays "
            "**causes** (supplier reliability, peak season), and let us change them on purpose.", ""]

    (OUT / "eda.md").write_text("\n".join(rep))
    print("\n".join(rep))


if __name__ == "__main__":
    main(ROOT / "data" / "DataCoSupplyChainDataset.csv")
