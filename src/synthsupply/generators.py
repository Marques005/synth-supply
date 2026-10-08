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


class GaussianCopula:
    """Learned generator: keeps every column's distribution AND the correlations.

    Three steps, each reversible:
      1. column -> uniform : pass each value through its column's CDF
                             (a value at the 30th percentile becomes 0.30)
      2. uniform -> normal : apply the inverse normal CDF (0.30 -> -0.52)
      3. learn the dependence: in this "normal space", all relationships
                             between columns are summarised by ONE correlation matrix.
    To sample: draw from a multivariate normal with that matrix, then walk
    back (normal -> uniform -> original values).

    Categorical columns: each category gets a slice of (0, 1) proportional to
    its frequency, e.g. "Standard Class" (60%) -> [0, 0.60).
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    # ---- step 1 and its inverse, per column --------------------------------
    def _to_uniform(self, s: pd.Series) -> np.ndarray:
        n = len(s)
        if s.name in self.cat_slices:
            lo, hi = self.cat_slices[s.name]
            a, b = s.map(lo).to_numpy(), s.map(hi).to_numpy()
            return a + self.rng.random(n) * (b - a)          # random point inside its slice
        # numeric: empirical CDF. Many rows share the same value (e.g. quantity = 1),
        # so each one gets a random point inside the block of ranks its value occupies.
        x = s.to_numpy()
        first_rank = stats.rankdata(x, method="min") - 1
        last_rank = stats.rankdata(x, method="max")
        return (first_rank + self.rng.random(n) * (last_rank - first_rank)) / n

    def _from_uniform(self, col: str, u: np.ndarray) -> np.ndarray:
        if col in self.cat_slices:
            cats, edges = self.cat_edges[col]
            idx = np.searchsorted(edges, u, side="right") - 1  # which slice does u fall in?
            return cats[np.clip(idx, 0, len(cats) - 1)]
        return np.quantile(self.values[col], u, method="inverted_cdf")  # inverse empirical CDF

    # ---- public API ---------------------------------------------------------
    def fit(self, df: pd.DataFrame):
        self.cat_slices, self.cat_edges, self.values = {}, {}, {}
        for c in COLUMNS:
            if c in CATEGORICAL:
                freq = df[c].value_counts(normalize=True)
                edges = np.concatenate([[0.0], freq.cumsum().to_numpy()])
                self.cat_slices[c] = (dict(zip(freq.index, edges[:-1])), dict(zip(freq.index, edges[1:])))
                self.cat_edges[c] = (freq.index.to_numpy(), edges)
            else:
                self.values[c] = df[c].to_numpy()

        eps = 1e-6  # keep u strictly inside (0, 1): the inverse normal of 0 or 1 is infinite
        z = np.column_stack([stats.norm.ppf(np.clip(self._to_uniform(df[c]), eps, 1 - eps)) for c in COLUMNS])
        self.corr = np.corrcoef(z, rowvar=False)            # step 3: the whole model is this matrix
        return self

    def sample(self, n: int) -> pd.DataFrame:
        z = self.rng.multivariate_normal(np.zeros(len(COLUMNS)), self.corr, size=n)
        u = stats.norm.cdf(z)
        return pd.DataFrame({c: self._from_uniform(c, u[:, i]) for i, c in enumerate(COLUMNS)})
