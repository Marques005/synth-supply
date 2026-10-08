"""Synthetic order-line generators.

Every generator has the same interface:
    gen = SomeGenerator(seed=42).fit(train_df)   # learn from real data
    synthetic_df = gen.sample(n)                  # produce n new rows
"""

import numpy as np
import pandas as pd
from scipy import stats

from .prepare import CATEGORICAL, COLUMNS


class IndependentMarginals:
    """Baseline: sample every column on its own.

    Each column keeps its exact real distribution (its "marginal"), but all
    relationships between columns are destroyed. Any serious generator must
    beat this.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, df: pd.DataFrame):
        self.columns = {c: df[c].to_numpy() for c in COLUMNS}
        return self

    def sample(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({
            c: self.rng.choice(values, size=n, replace=True) for c, values in self.columns.items()
        })
