"""How good is a synthetic dataset? Three questions, three groups of metrics.

FIDELITY - does it look like the real data?
    column_shapes : per column, 1 - KS (numeric) or 1 - TVD (categorical)
    column_pairs  : per pair of columns, 1 - TVD of their joint distribution
    rule_validity : share of rows that respect the business rules
UTILITY - is it useful for training models?
    TSTR : Train on Synthetic, Test on Real holdout (AUC for "late")
    TRTR : Train on Real, Test on Real holdout (the ceiling to compare with)
PRIVACY - does it leak the real rows it learned from?
    exact_copy_rate : share of rows identical to a training row, compared with the same
                      share for real holdout rows (the "natural" rate of coincidences)
    median_dcr      : median distance to the closest training record
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .prepare import CATEGORICAL, COLUMNS, NUMERIC, TARGET

PROMISE = {"Same Day": 0, "First Class": 1, "Second Class": 2, "Standard Class": 4}


# ----------------------------------------------------------------- fidelity --
def _tvd(a: pd.Series, b: pd.Series) -> float:
    """Total variation distance: half the sum of the differences in proportions."""
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    return 0.5 * pa.subtract(pb, fill_value=0).abs().sum()


def column_shapes(real: pd.DataFrame, synth: pd.DataFrame) -> pd.Series:
    out = {}
    for c in COLUMNS:
        if c in NUMERIC:
            out[c] = 1 - stats.ks_2samp(real[c], synth[c]).statistic
        else:
            out[c] = 1 - _tvd(real[c], synth[c])
    return pd.Series(out)


def _binned(real: pd.DataFrame, synth: pd.DataFrame, c: str, bins: int = 10):
    """Turn a column into labels so joint distributions can be compared with TVD.
    Columns with few values are used as they are; continuous ones are cut into deciles."""
    if c in CATEGORICAL or real[c].nunique() <= bins:
        return real[c].astype(str), synth[c].astype(str)
    edges = np.unique(np.quantile(real[c], np.linspace(0, 1, bins + 1)))
    def cut(s):
        return pd.cut(s.clip(edges[0], edges[-1]), edges, include_lowest=True).astype(str)
    return cut(real[c]), cut(synth[c])


def column_pairs(real: pd.DataFrame, synth: pd.DataFrame) -> pd.Series:
    binned = {c: _binned(real, synth, c) for c in COLUMNS}
    out = {}
    for a, b in combinations(COLUMNS, 2):                      # 13 columns -> 78 pairs
        joint_real = binned[a][0] + "|" + binned[b][0]
        joint_synth = binned[a][1] + "|" + binned[b][1]
        out[f"{a} x {b}"] = 1 - _tvd(joint_real, joint_synth)
    return pd.Series(out)


def rule_validity(df: pd.DataFrame) -> dict:
    promise_ok = df.days_scheduled == df.shipping_mode.map(PROMISE)
    late_ok = df[TARGET] == (df.days_real > df.days_scheduled).astype(int)
    return {"rule_promise": promise_ok.mean(), "rule_late": late_ok.mean(),
            "rule_all": (promise_ok & late_ok).mean()}


# ------------------------------------------------------------------ utility --
# days_real is excluded: it reveals the answer (late = days_real > days_scheduled).
FEATURES = [c for c in COLUMNS if c not in ("days_real", TARGET)]


def auc_on_holdout(train: pd.DataFrame, holdout: pd.DataFrame, seed: int = 0) -> float:
    """Train a late-delivery classifier on `train`, report its AUC on the real holdout."""
    cats = {c: sorted(set(train[c]) | set(holdout[c])) for c in CATEGORICAL}

    def encode(df):
        X = df[FEATURES].copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c], categories=cats[c])
        return X, df[TARGET].astype(int)

    X_train, y_train = encode(train)
    X_test, y_test = encode(holdout)
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=seed)
    model.fit(X_train, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


# ------------------------------------------------------------------ privacy --
def exact_copy_rate(train: pd.DataFrame, other: pd.DataFrame) -> float:
    seen = set(map(tuple, train[COLUMNS].astype(str).to_numpy()))
    return float(np.mean([tuple(r) in seen for r in other[COLUMNS].astype(str).to_numpy()]))


def _encode_for_distance(ref: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Numeric columns standardised with ref's mean/std; categorical ones one-hot encoded."""
    num = (df[NUMERIC] - ref[NUMERIC].mean()) / ref[NUMERIC].std()
    cat = pd.get_dummies(df[CATEGORICAL], dtype=float)
    cat = cat.reindex(columns=pd.get_dummies(ref[CATEGORICAL], dtype=float).columns, fill_value=0.0)
    return np.hstack([num.to_numpy(), cat.to_numpy()])


def median_dcr(train: pd.DataFrame, other: pd.DataFrame, n: int = 5000, seed: int = 0) -> float:
    ref = train.sample(min(len(train), 50_000), random_state=seed)   # subsample to keep it fast
    query = other.sample(min(len(other), n), random_state=seed)
    nn = NearestNeighbors(n_neighbors=1).fit(_encode_for_distance(ref, ref))
    dist, _ = nn.kneighbors(_encode_for_distance(ref, query))
    return float(np.median(dist))
