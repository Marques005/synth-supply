"""Quick sanity check of one generator against the real holdout.

Usage:  python scripts/check_generator.py independent
        python scripts/check_generator.py copula

Not the final evaluation (that comes in step 4): just enough to see, with our
own eyes, what a generator keeps and what it breaks.
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synthsupply import generators as G  # noqa: E402
from synthsupply.prepare import COLUMNS, NUMERIC, load_reference, split  # noqa: E402

GENERATORS = {
    "independent": lambda: G.IndependentMarginals(seed=42),
    "copula": lambda: G.GaussianCopula(seed=42),
}
MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]


def main(name: str):
    real = load_reference(ROOT / "data" / "DataCoSupplyChainDataset.csv")
    train, holdout = split(real)
    synth = GENERATORS[name]().fit(train).sample(len(train))

    print(f"\n=== {name}: {len(synth):,} synthetic rows ===\n")
    print(synth.head(), "\n")

    # 1. Does each column, on its own, look like the real one?
    sim = {}
    for c in COLUMNS:
        if c in NUMERIC:
            sim[c] = 1 - stats.ks_2samp(holdout[c], synth[c]).statistic
        else:
            p, q = holdout[c].value_counts(normalize=True), synth[c].value_counts(normalize=True)
            sim[c] = 1 - 0.5 * p.subtract(q, fill_value=0).abs().sum()
    print("1) Per-column similarity (1 = identical distribution):")
    print(pd.Series(sim).round(3).to_string(), "\n")

    # 2. Are the business rules respected?
    promise_ok = synth.days_scheduled == synth.shipping_mode.map(
        holdout.groupby("shipping_mode").days_scheduled.first())
    late_ok = synth.late == (synth.days_real > synth.days_scheduled).astype(int)
    print("2) Business rules:")
    print(f"   rows with the right promise for their mode : {promise_ok.mean():.1%}")
    print(f"   rows where late == (real > promise)         : {late_ok.mean():.1%}")
    print(f"   rows that respect both rules                : {(promise_ok & late_ok).mean():.1%}\n")

    # 3. Is the key relationship (late rate by mode) kept?
    comp = pd.DataFrame({
        "real": holdout.groupby("shipping_mode").late.mean(),
        "synthetic": synth.groupby("shipping_mode").late.mean(),
    }).loc[MODES]
    print("3) Late rate by shipping mode:")
    print(comp.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "independent")
