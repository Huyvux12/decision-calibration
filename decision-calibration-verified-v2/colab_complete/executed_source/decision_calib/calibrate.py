"""Temperature scaling for decision-model distributions.

We only observe probabilities, not logits. Temperature scaling on
log-probabilities is equivalent up to the (irrelevant) additive constant:

    p_T(k) = softmax(log(p(k) + eps) / T)

Fit modes (mirroring RQ2 of arXiv:2610.00346, which fitted per-task
temperatures on held-out data):
    "global"   - one T for all rows
    "per_type" - one T per question type (noul / choice / score)
    "per_task" - one T per (domain, question type) cell

Fitting minimizes NLL on the fit rows (scipy bounded scalar search).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from .metrics import EPS


def logits_from_probs(P):
    return np.log(np.clip(np.asarray(P, dtype=float), EPS, 1.0))


def apply_temperature(P, T):
    T = max(float(T), 1e-3)
    return softmax(logits_from_probs(P) / T, axis=-1)


def _nll_for_T(T, P_list, gold_idx):
    nlls = []
    for pr, gi in zip(P_list, gold_idx):
        pt = apply_temperature(pr[None, :], T)[0]
        nlls.append(-np.log(max(pt[gi], EPS)))
    return float(np.mean(nlls))


def fit_temperature(rows, group_key=None, T_bounds=(0.05, 10.0)):
    """Fit temperature(s) on `rows` (each row needs probs, gold_idx, and the
    grouping attribute). Returns {group_value: T}."""
    P_list = [np.asarray(r["probs"], dtype=float) for r in rows]
    g = [int(r["gold_idx"]) for r in rows]
    groups = {}
    for i, r in enumerate(rows):
        key = group_key(r) if group_key else "all"
        groups.setdefault(key, []).append(i)
    out = {}
    for key, idx in groups.items():
        if len(idx) < 5:
            out[key] = 1.0  # too few rows: keep as-shipped
            continue
        Pg = [P_list[i] for i in idx]
        gg = [g[i] for i in idx]
        res = minimize_scalar(_nll_for_T, bounds=T_bounds, method="bounded",
                              args=(Pg, gg),
                              options={"xatol": 1e-4})
        out[key] = float(res.x)
    return out


def rescale_rows(rows, T_map, group_key):
    """Return new rows with probabilities divided through their group's T."""
    new_rows = []
    for r in rows:
        T = T_map.get(group_key(r), 1.0)
        P = apply_temperature(np.asarray(r["probs"], dtype=float)[None, :], T)[0]
        nr = dict(r)
        nr["probs"] = P
        nr["pred_idx"] = int(np.argmax(P))
        new_rows.append(nr)
    return new_rows


GROUPERS = {
    "global": lambda r: "all",
    "per_type": lambda r: r["qtype"],
    "per_task": lambda r: (r["domain"], r["qtype"]),
}
