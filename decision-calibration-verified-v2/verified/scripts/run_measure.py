"""Run on the Colab VM (GPU). Measures full probability distributions for one
registered model over all datasets, writing data/predictions_<model>.csv
on the VM (pull back with `colab download`).

This file is a TEMPLATE: scripts/build_bundle.py inlines
decision_calib/models.py + decision_calib/datasets.py and bakes in MODEL,
producing dist/measure_<model>.py -- the single file actually executed.

Usage on the VM:
    python measure_<model>.py [--probe-only] [--out PATH]
"""
import sys, os, json, csv, time, traceback

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

MODEL_NAME = "@MODEL@"          # baked in by the build step
PROBE_ONLY = "--probe-only" in sys.argv
if "--help" in sys.argv or "-h" in sys.argv:
    print(__doc__)
    print("Options:\n"
          "  --probe-only   run 6 sample cases per loader and print raw answer keys\n"
          "  --out PATH     output CSV path (default: /tmp/predictions_<model>.csv)")
    sys.exit(0)
OUT = None
for i, a in enumerate(sys.argv):
    if a == "--out" and i + 1 < len(sys.argv):
        OUT = sys.argv[i + 1]
if OUT is None:
    OUT = f"/tmp/predictions_{MODEL_NAME}.csv"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decision_calib import models, datasets


def log(*a):
    print("LOG", *a, flush=True)


def main():
    t0 = time.time()
    entry, handle = models.load_model(MODEL_NAME)
    log(f"model {MODEL_NAME} ready in {time.time()-t0:.0f}s")
    normalize = entry["normalize"]
    chunk = entry["chunk"]

    loaders = [
        ("typed", lambda: datasets.typed_decisions()),
        ("ood", lambda: (datasets.ood_agnews() + datasets.ood_rotten_balanced()
                         + datasets.ood_sst2())),
    ]

    if PROBE_ONLY:
        print("PROBEBEGIN", flush=True)
        for lname, fn in loaders:
            cases = fn()[:6]
            items = [(c["state"], c["questions"]) for c in cases]
            raw = entry["predict_batch"](handle, items, chunk=min(chunk, 6))
            for c, r in zip(cases, raw):
                for qn, g in c["gold"].items():
                    print("PROBE", json.dumps({
                        "domain": c["domain"], "qname": qn,
                        "qtype": g["type"], "gold": g["label"],
                        "raw_keys": sorted(r[qn].keys()),
                    }), flush=True)
                    break
                if c["domain"].startswith("typed_"):
                    break
        print("PROBEEND", flush=True)
        return

    n_rows, n_warn = 0, 0
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "domain", "row_id", "question", "qtype",
                    "gold_idx", "pred_idx", "labels_json", "probs_json"])
        for lname, fn in loaders:
            for c in fn():
                items = [(c["state"], c["questions"])]
                try:
                    raw = entry["predict_batch"](handle, items, chunk=1)[0]
                except Exception:
                    log("PREDICT-ERROR", c["domain"], c["row_id"])
                    traceback.print_exc()
                    continue
                for qn, g in c["gold"].items():
                    qt, gold_label = g["type"], g["label"]
                    try:
                        norm = normalize(raw[qn], qt)
                    except Exception:
                        log("NORMALIZE-ERROR", c["domain"], c["row_id"], qn, qt)
                        continue
                    labels = sorted(norm["probs"].keys())
                    if gold_label not in norm["probs"]:
                        n_warn += 1
                        if n_warn <= 5:
                            log("GOLD-MISMATCH", c["domain"], c["row_id"], qn,
                                qt, repr(gold_label), repr(labels[:12]))
                        continue
                    probs = [norm["probs"][lb] for lb in labels]
                    gold_idx = labels.index(gold_label)
                    import numpy as np
                    pred_idx = (labels.index(norm["pred"])
                                if norm["pred"] in labels
                                else int(np.argmax(probs)))
                    w.writerow([MODEL_NAME, c["domain"], c["row_id"], qn, qt,
                                gold_idx, pred_idx,
                                json.dumps(labels, ensure_ascii=False),
                                json.dumps([round(float(p), 6)
                                            for p in probs])])
                    n_rows += 1
            log(f"loader {lname}: cumulative rows={n_rows} warnings={n_warn}")
    log(f"DONE rows={n_rows} warnings={n_warn} elapsed={time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
