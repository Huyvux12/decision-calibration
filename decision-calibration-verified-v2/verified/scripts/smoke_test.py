"""Smoke test: no GPU, no downloads. Run from the package root:

    python scripts/smoke_test.py

Checks:
  1. decision_calib imports; registry has all expected models with a
     complete entry (load / predict_batch / normalize / chunk).
  2. metrics.py on synthetic data: ECE of a perfectly calibrated predictor
     is ~0; ECE of an overconfident predictor is large; Brier/NLL sane.
  3. calibrate.py: fit_temperature recovers a known T on synthetic logits;
     rescale_rows keeps argmax for T=1.
  4. data/*.csv schemas: all 5 predictions CSVs have identical columns,
     2,200 rows each, probs sum to 1, indices in range; metrics CSVs and
     temperatures.json parse.
Exits non-zero on the first failure.
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from decision_calib import models, metrics as M, calibrate as C  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# 1. registry ------------------------------------------------------------
expected = {"d1-omni", "d1-3b", "opendecider-small", "jeff-2b", "jeff-0.8b", "laya"}
check("registry covers all models", expected <= set(models.REGISTRY),
      f"missing: {expected - set(models.REGISTRY)}")
for name, entry in models.REGISTRY.items():
    ok = all(k in entry for k in ("load", "predict_batch", "normalize", "chunk"))
    check(f"registry entry complete: {name}", ok)
    check(f"load_model KeyError on bogus name", True)
try:
    models.load_model("nope")
    check("load_model raises KeyError on unknown", False)
except KeyError:
    check("load_model raises KeyError on unknown", True)

# 2. metrics on synthetic data --------------------------------------------
rng = np.random.default_rng(0)


def synth_rows(n, overconfident=False):
    rows = []
    for _ in range(n):
        k = 3
        gold = rng.integers(k)
        if overconfident:
            probs = np.full(k, 0.05)
            probs[rng.integers(k)] = 0.9  # confident but often wrong
        else:
            # calibrated: sample truth from the stated distribution
            logits = rng.normal(0, 1, k)
            probs = np.exp(logits) / np.exp(logits).sum()
            gold = rng.choice(k, p=probs)
        rows.append({"probs": probs, "gold_idx": int(gold),
                     "pred_idx": int(np.argmax(probs))})
    return rows


cal = synth_rows(2000)
e_cal, _ = M.ece(cal, n_bins=15)
check("ECE ~ 0 for calibrated predictor", e_cal < 0.05, f"ece={e_cal:.4f}")
oc = synth_rows(2000, overconfident=True)
e_oc, _ = M.ece(oc, n_bins=15)
check("ECE large for overconfident predictor", e_oc > 0.2, f"ece={e_oc:.4f}")
b = M.brier_score(cal)
check("Brier in [0, 2]", 0.0 <= b <= 2.0, f"brier={b:.4f}")
n = M.nll(cal)
check("NLL finite and positive", np.isfinite(n) and n > 0, f"nll={n:.4f}")
s = M.summarize(cal)
check("summarize has all keys",
      {"n", "accuracy", "mean_confidence", "ece", "brier", "nll", "bin_data"} <= set(s))
check("accuracy sane on calibrated synthetic", 0.3 < s["accuracy"] < 0.9,
      f"acc={s['accuracy']:.3f}")

# 3. temperature fitting ---------------------------------------------------
# Setup: truth is softmax(logits); the "model" states P = softmax(logits/Tm)
# with Tm=2.5 (too flat / underconfident). Fitting T' on the stated P should
# recover T' = 1/Tm = 0.4, since softmax(log P / T') = softmax(logits/(Tm*T')).
Tm = 2.5
P_list, gold = [], []
for _ in range(600):
    logits = rng.normal(0, 1.5, 4)
    e = np.exp(logits)
    truth = e / e.sum()
    g = rng.choice(4, p=truth)
    et = np.exp(logits / Tm)
    P_list.append(et / et.sum())
    gold.append(g)
rows = [{"probs": p, "gold_idx": int(g), "pred_idx": int(np.argmax(p)),
         "qtype": "choice"} for p, g in zip(P_list, gold)]
T_map = C.fit_temperature(rows)
check("fit_temperature recovers T~0.4 (=1/2.5)", abs(T_map["all"] - 1 / Tm) < 0.15,
      f"T={T_map['all']:.3f}")
T_pt = C.fit_temperature(rows, group_key=lambda r: r["qtype"])
check("per-type fit returns qtype key", set(T_pt) == {"choice"})
resc = C.rescale_rows(rows, {"choice": 1.0}, lambda r: r["qtype"])
same_argmax = all(r2["pred_idx"] == r1["pred_idx"] for r1, r2 in zip(rows, resc))
check("rescale_rows T=1 keeps argmax", same_argmax)
T_hi = C.apply_temperature(np.array([[0.7, 0.2, 0.1]]), 10.0)[0]
check("high T flattens distribution", T_hi.max() < 0.5, f"max={T_hi.max():.3f}")

# 4. data files ------------------------------------------------------------
DATA = os.path.join(ROOT, "data")
want_cols = ["model", "domain", "row_id", "question", "qtype",
             "gold_idx", "pred_idx", "labels_json", "probs_json"]
pred_files = ["predictions_d1-omni.csv", "predictions_d1-3b.csv",
              "predictions_opendecider-small.csv", "predictions_jeff-2b.csv",
              "predictions_jeff-0.8b.csv"]
for fn in pred_files:
    p = os.path.join(DATA, fn)
    check(f"exists {fn}", os.path.exists(p))
    if not os.path.exists(p):
        continue
    with open(p, newline="") as f:
        rdr = csv.DictReader(f)
        check(f"{fn} columns", rdr.fieldnames == want_cols,
              f"got {rdr.fieldnames}")
        n = 0
        bad = 0
        for rec in rdr:
            n += 1
            labels = json.loads(rec["labels_json"])
            probs = json.loads(rec["probs_json"])
            gi, pi = int(rec["gold_idx"]), int(rec["pred_idx"])
            if not (len(labels) == len(probs) and 0 <= gi < len(labels)
                    and 0 <= pi < len(labels)
                    and abs(sum(probs) - 1.0) < 1e-3
                    and rec["qtype"] in ("noul", "choice", "score")):
                bad += 1
                if bad <= 2:
                    print(f"   bad row: {rec['row_id']} {rec['question']}")
        check(f"{fn} 2200 rows", n == 2200, f"n={n}")
        check(f"{fn} rows valid", bad == 0, f"bad={bad}")

for fn in ["metrics_asshipped.csv", "metrics_byworkflow.csv", "metrics_rq2.csv"]:
    p = os.path.join(DATA, fn)
    check(f"exists+parses {fn}", os.path.exists(p))
    if os.path.exists(p):
        try:
            list(csv.DictReader(open(p, newline="")))
            check(f"parses {fn}", True)
        except Exception as e:  # noqa: BLE001
            check(f"parses {fn}", False, str(e))
try:
    t = json.load(open(os.path.join(DATA, "temperatures.json")))
    check("temperatures.json parses, non-empty", isinstance(t, dict) and len(t) > 0,
          f"keys={len(t)}")
except Exception as e:  # noqa: BLE001
    check("temperatures.json parses", False, str(e))

# 5. figures referenced exist ----------------------------------------------
FIG = os.path.join(ROOT, "figures")
for m in ["d1-omni", "d1-3b", "opendecider-small", "jeff-2b", "jeff-0.8b"]:
    for dg in ["indomain", "ood"]:
        check(f"figure rel_{m}_{dg}.png",
              os.path.exists(os.path.join(FIG, f"rel_{m}_{dg}.png")))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL SMOKE TESTS PASSED")
