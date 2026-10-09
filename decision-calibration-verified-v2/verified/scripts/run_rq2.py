"""Local analysis (no GPU): as-shipped calibration metrics + RQ2 replication.

Reads:  data/predictions_<model>.csv   (from scripts/run_measure.py)
Writes: data/metrics_asshipped.csv     - per (model, domain_group, qtype)
        data/metrics_byworkflow.csv    - per (model, workflow) in-domain
        data/metrics_rq2.csv           - temperature-fitting experiments
        data/temperatures.json         - fitted T values

RQ2 design (mirrors arXiv:2610.00346 RQ2: per-task temperature on held-out):
  fit sets:
    A. in-domain held-out 20% (stratified by domain+qtype), seed fixed
    B. OOD 50% (stratified), seed fixed            [small-n caveat]
  eval sets:
    A1. in-domain remainder (same distribution as fit)
    A2. ALL OOD rows  (transfer: does in-domain-fitted T help OOD?)
    B1. OOD remainder
  modes: global / per_type / per_task(=(domain,qtype))
"""
import os, sys, json, csv, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decision_calib import metrics as M
from decision_calib import calibrate as C

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SEED = 20261008
N_BINS = 15
DEFAULT_MODELS = ["d1-omni", "d1-3b", "opendecider-small", "jeff-2b", "jeff-0.8b"]


def load_rows(model):
    path = os.path.join(DATA, f"predictions_{model}.csv")
    rows = []
    with open(path, newline="") as f:
        for rec in csv.DictReader(f):
            labels = json.loads(rec["labels_json"])
            probs = np.array(json.loads(rec["probs_json"]), dtype=float)
            rows.append({
                "model": rec["model"], "domain": rec["domain"],
                "row_id": rec["row_id"], "question": rec["question"],
                "qtype": rec["qtype"], "labels": labels,
                "probs": probs,
                "gold_idx": int(rec["gold_idx"]),
                "pred_idx": int(rec["pred_idx"]),
            })
    return rows


def domain_group(domain):
    return "in-domain" if domain.startswith("typed_") else "ood"


def summarize_block(rows):
    s = M.summarize(rows, n_bins=N_BINS)
    s.pop("bin_data", None)
    return s


def stratified_split(rows, frac, seed):
    """Deterministic stratified split by (domain, qtype). Returns (fit, rest)."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["domain"], r["qtype"]), []).append(i)
    fit_idx = set()
    for key, idx in groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * frac)))
        fit_idx.update(idx[:k].tolist())
    fit = [r for i, r in enumerate(rows) if i in fit_idx]
    rest = [r for i, r in enumerate(rows) if i not in fit_idx]
    return fit, rest


def main():
    ap = argparse.ArgumentParser(
        description="As-shipped calibration metrics + RQ2 temperature-fitting "
                    "experiments. Reads data/predictions_<model>.csv, writes "
                    "metrics_asshipped.csv, metrics_byworkflow.csv, "
                    "metrics_rq2.csv and temperatures.json (all in data/).")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="registered model names to analyze "
                         f"(default: {' '.join(DEFAULT_MODELS)})")
    models = ap.parse_args().models
    all_rows = {m: load_rows(m) for m in models}
    for m, rows in all_rows.items():
        print(f"{m}: {len(rows)} rows", flush=True)
        for r in rows:
            r["dgroup"] = domain_group(r["domain"])

    # ---- 1. as-shipped metrics per (model, dgroup, qtype)
    with open(os.path.join(DATA, "metrics_asshipped.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dgroup", "qtype", "n", "accuracy",
                    "mean_confidence", "ece", "brier", "nll"])
        for m in models:
            for dg in ["in-domain", "ood"]:
                for qt in ["noul", "choice", "score"]:
                    rows = [r for r in all_rows[m]
                            if r["dgroup"] == dg and r["qtype"] == qt]
                    s = summarize_block(rows)
                    w.writerow([m, dg, qt, s["n"],
                                f"{s['accuracy']:.4f}", f"{s['mean_confidence']:.4f}",
                                f"{s['ece']:.4f}", f"{s['brier']:.4f}",
                                f"{s['nll']:.4f}"])

    # ---- 2. per-workflow in-domain (accuracy + ece)
    with open(os.path.join(DATA, "metrics_byworkflow.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "workflow", "n", "accuracy", "ece", "brier"])
        for m in models:
            wfs = sorted({r["domain"] for r in all_rows[m]
                          if r["dgroup"] == "in-domain"})
            for wf in wfs:
                rows = [r for r in all_rows[m] if r["domain"] == wf]
                s = summarize_block(rows)
                w.writerow([m, wf.replace("typed_", ""), s["n"],
                            f"{s['accuracy']:.4f}", f"{s['ece']:.4f}",
                            f"{s['brier']:.4f}"])

    # ---- 3. RQ2 temperature experiments
    temperatures = {}
    rq2_out = []
    for m in models:
        rows = all_rows[m]
        in_dom = [r for r in rows if r["dgroup"] == "in-domain"]
        ood = [r for r in rows if r["dgroup"] == "ood"]

        experiments = []
        # A: fit on in-domain held-out 20%
        fitA, restA = stratified_split(in_dom, 0.20, SEED)
        experiments.append(("fit_indom20", fitA,
                            [("eval_indom_rest", restA),
                             ("eval_ood_all", ood)]))
        # B: fit on OOD 50%
        fitB, restB = stratified_split(ood, 0.50, SEED + 1)
        experiments.append(("fit_ood50", fitB, [("eval_ood_rest", restB)]))

        for fit_name, fit_rows, evals in experiments:
            for mode, grouper in C.GROUPERS.items():
                T_map = C.fit_temperature(fit_rows, group_key=grouper)
                temperatures[f"{m}/{fit_name}/{mode}"] = {
                    str(k): round(float(v), 4) for k, v in T_map.items()}
                for eval_name, eval_rows in evals:
                    resc = C.rescale_rows(eval_rows, T_map, grouper)
                    s = summarize_block(resc)
                    rq2_out.append({
                        "model": m, "fit": fit_name, "mode": mode,
                        "eval": eval_name, "n_fit": len(fit_rows),
                        "n_eval": len(eval_rows), **{k: s[k] for k in
                            ("accuracy", "mean_confidence", "ece",
                             "brier", "nll")}})
        # as-shipped baselines on the same eval sets for comparison
        for eval_name, eval_rows in [("eval_indom_rest", restA),
                                    ("eval_ood_all", ood),
                                    ("eval_ood_rest", restB)]:
            s = summarize_block(eval_rows)
            rq2_out.append({"model": m, "fit": "-", "mode": "as-shipped",
                            "eval": eval_name, "n_fit": 0,
                            "n_eval": len(eval_rows), **{k: s[k] for k in
                            ("accuracy", "mean_confidence", "ece",
                             "brier", "nll")}})

    with open(os.path.join(DATA, "metrics_rq2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "fit", "mode", "eval",
                                          "n_fit", "n_eval", "accuracy",
                                          "mean_confidence", "ece", "brier",
                                          "nll"])
        w.writeheader()
        for rec in rq2_out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in rec.items()})
    with open(os.path.join(DATA, "temperatures.json"), "w") as f:
        json.dump(temperatures, f, indent=2)
    print(f"RQ2: {len(rq2_out)} eval rows written", flush=True)


if __name__ == "__main__":
    main()
