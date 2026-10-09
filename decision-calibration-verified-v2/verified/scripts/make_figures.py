"""Local (no GPU): reliability diagrams per (model, domain-group).

One figure per model x domain-group: 1x3 panels for noul / choice / score.
Each panel: reliability curve (equal-mass bins, 15), diagonal, ECE annotation.
Reads data/predictions_<model>.csv ; writes figures/rel_<model>_<dgroup>.png
"""
import os, sys, csv, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from decision_calib import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
N_BINS = 15
DEFAULT_MODELS = ["d1-omni", "d1-3b", "opendecider-small", "jeff-2b", "jeff-0.8b"]


def load_rows(model):
    rows = []
    with open(os.path.join(DATA, f"predictions_{model}.csv"), newline="") as f:
        for rec in csv.DictReader(f):
            rows.append({
                "qtype": rec["qtype"],
                "dgroup": ("in-domain" if rec["domain"].startswith("typed_")
                           else "ood"),
                "probs": np.array(json.loads(rec["probs_json"]), dtype=float),
                "gold_idx": int(rec["gold_idx"]),
                "pred_idx": int(rec["pred_idx"]),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Reliability diagrams per (model, domain-group): 1x3 "
                    "panels (noul/choice/score). Reads data/predictions_<model>.csv, "
                    "writes figures/rel_<model>_<dgroup>.png.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="registered model names to plot "
                         f"(default: {' '.join(DEFAULT_MODELS)})")
    for model in ap.parse_args().models:
        rows = load_rows(model)
        for dg in ["in-domain", "ood"]:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
            for ax, qt in zip(axes, ["noul", "choice", "score"]):
                sub = [r for r in rows if r["dgroup"] == dg and r["qtype"] == qt]
                if not sub:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes)
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                    ax.set_xlabel("mean confidence"); ax.set_title(f"{qt} (n=0)")
                    continue
                e, bd = M.ece(sub, n_bins=N_BINS)
                xs = np.array(bd["conf"]); ys = np.array(bd["acc"])
                cnt = np.array(bd["count"])
                ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
                # marker size ~ bin count
                sizes = 20 + 180 * (cnt / cnt.max() if cnt.max() else 1)
                ax.scatter(xs, ys, s=sizes, alpha=0.7, zorder=3)
                ax.plot(xs, ys, lw=1.5, alpha=0.6)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xlabel("mean confidence"); ax.set_title(
                    f"{qt}  (n={len(sub)}, ECE={e:.3f})")
                ax.grid(alpha=0.3)
            axes[0].set_ylabel("accuracy")
            fig.suptitle(f"Reliability — {model} — {dg} (equal-mass, "
                         f"{N_BINS} bins; size ~ bin count)")
            fig.tight_layout()
            out = os.path.join(FIG, f"rel_{model}_{dg.replace('-', '')}.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
