"""Pure-numpy calibration metrics for decision-model predictions.

Row convention (matches predictions CSVs produced by scripts/run_measure.py):
    row = {"probs": np.array([...]),      # full distribution over options
           "gold_idx": int,               # index of gold label in probs
           "pred_idx": int}               # index of argmax prediction

All functions take lists/arrays of rows in this form.
"""

import numpy as np

EPS = 1e-12


def as_arrays(rows):
    """Per-row lists (option counts vary across questions, so no stacking)."""
    P = [np.asarray(r["probs"], dtype=float) for r in rows]
    g = [int(r["gold_idx"]) for r in rows]
    p = [int(r["pred_idx"]) for r in rows]
    return P, g, p


def accuracy(rows):
    _, g, p = as_arrays(rows)
    return float(np.mean(np.array(p) == np.array(g))) if rows else float("nan")


def mean_confidence(rows):
    P, _, p = as_arrays(rows)
    if not rows:
        return float("nan")
    return float(np.mean([pr[pi] for pr, pi in zip(P, p)]))


def brier_score(rows):
    """Multiclass Brier: mean over rows of sum_k (p_k - onehot_k)^2.

    Note: for binary (noul) questions this sums over both classes, so it is
    2x the textbook binary Brier (p-y)^2 and lives on [0, 2]. Comparisons
    are only meaningful within the same question type."""
    P, g, _ = as_arrays(rows)
    if not rows:
        return float("nan")
    vals = []
    for pr, gi in zip(P, g):
        oh = np.zeros_like(pr)
        oh[gi] = 1.0
        vals.append(float(np.sum((pr - oh) ** 2)))
    return float(np.mean(vals))


def nll(rows):
    """Mean negative log-likelihood of the gold label."""
    P, g, _ = as_arrays(rows)
    if not rows:
        return float("nan")
    return float(np.mean([-np.log(max(pr[gi], EPS)) for pr, gi in zip(P, g)]))


def ece(rows, n_bins=15):
    """Expected Calibration Error with equal-mass binning on top-1 confidence.

    Returns (ece, bin_data) where bin_data has per-bin accuracy, confidence
    and count — the inputs for reliability diagrams.

    Note: with heavily tied confidences, equal-mass binning splits ties
    arbitrarily and ECE can be unstable; model confidences in practice are
    continuous, where this is not an issue.
    """
    P, g, p = as_arrays(rows)
    n = len(rows)
    if n == 0:
        return float("nan"), {"acc": [], "conf": [], "count": [],
                              "edges": [], "n_bins": n_bins}
    conf = np.array([pr[pi] for pr, pi in zip(P, p)])
    correct = (np.array(p) == np.array(g)).astype(float)
    order = np.argsort(conf, kind="stable")
    # equal-mass bins; the last bin absorbs the remainder
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    acc_b, conf_b, cnt_b, edge_b = [], [], [], []
    for b in range(n_bins):
        idx = order[edges[b]:edges[b + 1]]
        if len(idx) == 0:
            continue
        acc_b.append(float(np.mean(correct[idx])))
        conf_b.append(float(np.mean(conf[idx])))
        cnt_b.append(int(len(idx)))
        edge_b.append((float(conf[idx[0]]), float(conf[idx[-1]])))
    ece_val = float(np.sum(np.abs(np.array(acc_b) - np.array(conf_b))
                           * np.array(cnt_b) / n))
    return ece_val, {"acc": acc_b, "conf": conf_b, "count": cnt_b,
                     "edges": edge_b, "n_bins": n_bins}


def summarize(rows, n_bins=15):
    """One dict with every headline metric for a row set."""
    e, bindata = ece(rows, n_bins=n_bins)
    return {
        "n": len(rows),
        "accuracy": accuracy(rows),
        "mean_confidence": mean_confidence(rows),
        "ece": e,
        "brier": brier_score(rows),
        "nll": nll(rows),
        "bin_data": bindata,
    }
