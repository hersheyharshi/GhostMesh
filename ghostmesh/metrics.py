"""Small, dependency-free statistics used by the experiments."""
from __future__ import annotations

import numpy as np


def auc(score, label) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties count half)."""
    score = np.asarray(score, float)
    label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=float)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2
        i = j + 1
    ranks[order] = r
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_ci(x, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0):
    """Mean and percentile-bootstrap (1 - alpha) confidence interval."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def calibration(p, y, bins: int = 10):
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def roc_curve(score, label, n: int = 200):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    th = np.unique(np.quantile(score, np.linspace(0, 1, n)))[::-1]
    P, N = label.sum(), (~label).sum()
    tpr = [(score[label] >= t).sum() / P for t in th]
    fpr = [(score[~label] >= t).sum() / N for t in th]
    return np.r_[0, fpr, 1], np.r_[0, tpr, 1]
