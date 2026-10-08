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


class DomainSimulator:
    """Rule-based generator: a calibrated story of how an order line is born and delivered.

    Story for one order line:
      1. when       : month and weekday, with the real calendar mix
      2. where      : market, which depends on the month (DataCo's time blocks)
      3. who / what : customer segment, department, payment type, quantity, discount
      4. how        : shipping mode, which depends on the customer segment
      5. price      : drawn from the department's real price list
      6. supplier   : each department has a few suppliers, each with a hidden reliability
      7. promise    : days_scheduled = fixed promise of the shipping mode (a rule)
      8. delivery   : days_real from the real per-mode distribution, tilted towards later
                      days when risk is high (bad supplier, Nov/Dec peak, congested market)
      9. late       : days_real > days_scheduled (a rule)

    Steps 1-5 copy what DataCo shows. Steps 6 and 8 add what DataCo lacks: CAUSES for
    delays, while keeping the overall delay rates close to the real ones.
    `supplier_shocks={"FAN-1": 2.0}` makes a supplier suddenly worse (used for drift in P3).
    """

    def __init__(self, n_suppliers_per_dept: int = 3, risk_strength: float = 1.0,
                 peak_effect: float = 0.8, supplier_shocks: dict | None = None, seed: int = 0):
        self.n_sup = n_suppliers_per_dept
        self.risk_strength = risk_strength
        self.peak_effect = peak_effect
        self.supplier_shocks = supplier_shocks or {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _dist(s: pd.Series):
        """Empirical distribution of a column: (values, probabilities)."""
        p = s.value_counts(normalize=True)
        return p.index.to_numpy(), p.to_numpy()

    def _pick(self, dist, n: int) -> np.ndarray:
        values, probs = dist
        return self.rng.choice(values, size=n, p=probs)

    def _pick_given(self, df: pd.DataFrame, parent: str, table: dict) -> np.ndarray:
        """Sample a child column whose distribution depends on a parent column."""
        out = np.empty(len(df), dtype=object)
        for value, idx in df.groupby(parent).indices.items():
            out[idx] = self._pick(table[value], len(idx))
        return out

    # ---- fit: calibrate the story on real data ------------------------------
    def fit(self, df: pd.DataFrame):
        d = self._dist
        self.month, self.dow = d(df.order_month), d(df.order_dow)
        self.segment, self.department = d(df.customer_segment), d(df.department)
        self.payment, self.quantity, self.discount = d(df.payment_type), d(df.quantity), d(df.discount_rate)
        self.market_given_month = {k: d(g) for k, g in df.groupby("order_month").market}
        self.mode_given_segment = {k: d(g) for k, g in df.groupby("customer_segment").shipping_mode}
        self.price_given_dept = {k: d(g) for k, g in df.groupby("department").unit_price}
        self.promise = df.groupby("shipping_mode").days_scheduled.first().to_dict()
        offset = df.days_real - df.days_scheduled                      # how many days after the promise
        self.offset_given_mode = {k: d(g) for k, g in offset.groupby(df.shipping_mode)}

        # Hidden causes. A separate RNG so they do not change when we sample more rows.
        hidden = np.random.default_rng(self.seed + 1)
        self.suppliers = {dept: [f"{dept[:3].upper()}-{k + 1}" for k in range(self.n_sup)]
                          for dept in self.department[0]}
        self.supplier_risk = {s: hidden.normal(0, 1.0) for ss in self.suppliers.values() for s in ss}
        self.market_risk = {m: hidden.normal(0, 0.3) for m in df.market.unique()}
        return self

    # ---- step 8: delivery with risk ------------------------------------------
    def _tilted_offsets(self, mode: str, risk: np.ndarray) -> np.ndarray:
        """Sample delivery offsets, moving probability towards later days when risk > 0.

        p_new(day) is proportional to p_real(day) * exp(strength * risk * z(day)),
        where z(day) is the standardised offset. risk = 0 gives back the real distribution.
        """
        values, probs = self.offset_given_mode[mode]
        if len(values) == 1:                       # e.g. First Class: always the same offset
            return np.full(len(risk), values[0])
        z = (values - values.mean()) / values.std()
        w = probs[None, :] * np.exp(self.risk_strength * risk[:, None] * z[None, :])
        w /= w.sum(axis=1, keepdims=True)           # one probability vector per row
        u = self.rng.random(len(risk))[:, None]
        idx = (u > w.cumsum(axis=1)).sum(axis=1)    # inverse-CDF sampling, row by row
        return values[np.clip(idx, 0, len(values) - 1)]

    # ---- sample: tell the story n times --------------------------------------
    def sample(self, n: int) -> pd.DataFrame:
        df = pd.DataFrame({
            "order_month": self._pick(self.month, n),                                      # 1
            "order_dow": self._pick(self.dow, n),
            "customer_segment": self._pick(self.segment, n),                               # 3
            "department": self._pick(self.department, n),
            "payment_type": self._pick(self.payment, n),
            "quantity": self._pick(self.quantity, n),
            "discount_rate": self._pick(self.discount, n),
        })
        df["market"] = self._pick_given(df, "order_month", self.market_given_month)       # 2
        df["shipping_mode"] = self._pick_given(df, "customer_segment", self.mode_given_segment)  # 4
        df["unit_price"] = self._pick_given(df, "department", self.price_given_dept).astype(float)  # 5
        df["supplier_id"] = [self.rng.choice(self.suppliers[d]) for d in df.department]    # 6
        df["days_scheduled"] = df.shipping_mode.map(self.promise)                          # 7

        risk = (df.supplier_id.map(self.supplier_risk)                                     # 8
                + df.market.map(self.market_risk)
                + self.peak_effect * df.order_month.isin([11, 12]))
        risk = (risk - risk.mean()) + df.supplier_id.map(self.supplier_shocks).fillna(0.0)
        risk = 0.5 * risk.to_numpy()
        df["days_real"] = 0
        for mode, idx in df.groupby("shipping_mode").indices.items():
            df.loc[df.index[idx], "days_real"] = df.days_scheduled.iloc[idx].to_numpy() + \
                self._tilted_offsets(mode, risk[idx])
        df["late"] = (df.days_real > df.days_scheduled).astype(int)                       # 9
        return df[COLUMNS + ["supplier_id"]]
