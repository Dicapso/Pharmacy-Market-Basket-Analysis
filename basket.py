"""Market basket analysis toolkit: cleaning, Apriori (from scratch, NumPy),
association rules with significance tests, and a rule-based recommender."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

DATA = Path(__file__).resolve().parent / "data"
GROSS_MARGIN = 0.35        # assumed OTC gross margin
UPTAKE = 0.08              # assumed share of "A without B" baskets that accept a prompt


# ------------------------------------------------------------------ loading
def load_raw():
    products = pd.read_csv(DATA / "products.csv")
    baskets = pd.read_csv(DATA / "baskets.csv", parse_dates=["ts"])
    lines = pd.read_csv(DATA / "basket_lines.csv")
    return products, baskets, lines


def clean(lines):
    """Drop return lines (qty < 0) and exact duplicate scan lines. Returns (clean, log)."""
    log = {"Raw lines": len(lines)}
    returns = lines.qty < 0
    log["Return lines (qty < 0) removed"] = int(returns.sum())
    out = lines[~returns]
    dups = out.duplicated()
    log["Exact duplicate lines removed"] = int(dups.sum())
    out = out[~dups]
    # a product appears once per basket after this step
    out = out.groupby(["basket_id", "product_id"], as_index=False).agg(qty=("qty", "sum"), unit_price=("unit_price", "first"))
    out["revenue"] = out.qty * out.unit_price
    log["Clean lines"] = len(out)
    return out, log


def one_hot(lines, basket_ids=None):
    """Boolean basket x product matrix (DataFrame)."""
    m = pd.crosstab(lines.basket_id, lines.product_id).astype(bool)
    if basket_ids is not None:
        m = m.reindex(basket_ids, fill_value=False)
    return m


# ------------------------------------------------------------------ apriori
def apriori(X: pd.DataFrame, min_support=0.005, max_len=3):
    """Level-wise Apriori. Returns DataFrame[itemset (tuple), support, n]."""
    cols = np.array(X.columns)
    A = X.to_numpy()
    N = len(A)
    sup1 = A.mean(axis=0)
    freq = {(i,): sup1[i] for i in np.flatnonzero(sup1 >= min_support)}
    level = sorted(freq)
    all_freq = dict(freq)
    k = 2
    while level and k <= max_len:
        prev = set(level)
        cands = set()
        for a, b in combinations(level, 2):          # join step: share first k-2 items
            if a[:-1] == b[:-1]:
                c = tuple(sorted(set(a) | set(b)))
                if all(s in prev for s in combinations(c, k - 1)):   # prune step
                    cands.add(c)
        level = []
        for c in sorted(cands):
            s = A[:, list(c)].all(axis=1).mean()
            if s >= min_support:
                all_freq[c] = s
                level.append(c)
        k += 1
    out = pd.DataFrame([(tuple(cols[list(i)]), s) for i, s in all_freq.items()], columns=["itemset", "support"])
    out["n"] = (out.support * N).round().astype(int)
    out["size"] = out.itemset.map(len)
    return out.sort_values(["size", "support"], ascending=[True, False]).reset_index(drop=True)


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def rules(freq: pd.DataFrame, N: int):
    """All association rules A -> B from frequent itemsets, with lift, leverage,
    conviction and a 2x2 chi-square test (BH-adjusted q-value)."""
    sup = dict(zip(freq.itemset.map(frozenset), freq.support))
    rows = []
    for items, s_ab in sup.items():
        if len(items) < 2:
            continue
        for r in range(1, len(items)):
            for ant in combinations(sorted(items), r):
                ant = frozenset(ant)
                con = items - ant
                s_a, s_b = sup[ant], sup[con]
                conf = s_ab / s_a
                lift = conf / s_b
                # 2x2 contingency chi-square (A present/absent x B present/absent)
                n11 = s_ab * N; n10 = (s_a - s_ab) * N; n01 = (s_b - s_ab) * N; n00 = N - n11 - n10 - n01
                exp = np.outer([s_a, 1 - s_a], [s_b, 1 - s_b]) * N
                obs = np.array([[n11, n10], [n01, n00]])
                stat = ((obs - exp) ** 2 / exp).sum()
                rows.append(dict(antecedent=tuple(sorted(ant)), consequent=tuple(sorted(con)),
                                 support=s_ab, ant_support=s_a, con_support=s_b, confidence=conf, lift=lift,
                                 leverage=s_ab - s_a * s_b,
                                 conviction=np.inf if conf >= 1 else (1 - s_b) / (1 - conf),
                                 chi2=stat, p_value=chi2.sf(stat, 1)))
    R = pd.DataFrame(rows)
    R["q_value"] = bh_fdr(R.p_value)
    return R.sort_values("lift", ascending=False).reset_index(drop=True)


def label(R, names):
    f = lambda t: " + ".join(names[i] for i in t)
    R = R.copy()
    R["rule"] = R.antecedent.map(f) + "  →  " + R.consequent.map(f)
    return R


def pair_lift(X: pd.DataFrame):
    """Pairwise co-occurrence lift for every product pair (includes lift < 1 = substitutes)."""
    A = X.to_numpy().astype(float)
    N = len(A)
    co = A.T @ A
    s = A.mean(axis=0)
    exp = np.outer(s, s) * N
    out = []
    for i, j in combinations(range(A.shape[1]), 2):
        out.append((X.columns[i], X.columns[j], co[i, j], exp[i, j], co[i, j] / exp[i, j]))
    return pd.DataFrame(out, columns=["a", "b", "observed", "expected", "lift"])


def stratified_pair_lift(X: pd.DataFrame, groups: pd.Series):
    """Lift with the expectation computed inside each stratum (e.g. month):
    Σ_g observed_g / Σ_g N_g·s_a,g·s_b,g. Removes associations created only by seasonality."""
    obs = exp = 0
    for _, g in X.groupby(groups.reindex(X.index)):
        A = g.to_numpy().astype(float)
        obs = obs + A.T @ A
        s = A.mean(axis=0)
        exp = exp + np.outer(s, s) * len(A)
    i, j = np.triu_indices(X.shape[1], k=1)
    return pd.DataFrame({"a": X.columns[i], "b": X.columns[j], "strat_lift": obs[i, j] / exp[i, j]})


# ------------------------------------------------------------------ recommender
class RuleRecommender:
    """Score unseen products by the best confidence among matching rules
    (antecedent ⊆ basket, single-item consequent). Falls back to popularity."""

    def __init__(self, R, popularity, exclude=()):
        R = R[(R.consequent.map(len) == 1) & (R.lift > 1)]
        self.rules = [(frozenset(a), c[0], conf, lift) for a, c, conf, lift in
                      zip(R.antecedent, R.consequent, R.confidence, R.lift) if c[0] not in exclude]
        self.pop = [p for p in popularity if p not in exclude]

    def recommend(self, basket, k=3):
        basket = set(basket)
        score = {}
        for ant, con, conf, lift in self.rules:
            if con not in basket and ant <= basket:
                if conf > score.get(con, (0, 0))[0]:
                    score[con] = (conf, lift)
        recs = [p for p, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)][:k]
        for p in self.pop:
            if len(recs) >= k:
                break
            if p not in basket and p not in recs:
                recs.append(p)
        return recs


def popularity_recs(basket, popularity, k=3):
    return [p for p in popularity if p not in basket][:k]


def hide_one_eval(test_baskets, recommender, popularity, k=3, seed=7):
    """Leave-one-out: hide one random item from each basket (>=2 items), check if it is in top-k."""
    rng = np.random.default_rng(seed)
    hits_rule = hits_pop = 0
    for items in test_baskets:
        items = list(items)
        hidden = items[rng.integers(len(items))]
        rest = [i for i in items if i != hidden]
        hits_rule += hidden in recommender.recommend(rest, k)
        hits_pop += hidden in popularity_recs(rest, popularity, k)
    n = len(test_baskets)
    return n, hits_rule / n, hits_pop / n
