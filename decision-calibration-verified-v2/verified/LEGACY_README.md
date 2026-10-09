# decision-calibration

Reusable harness for calibration studies of **zero-token decision models**
(System One / typed-decision models: they return probabilities over fixed
options with zero output tokens).

Built for the study *"Calibration of the Liquid d1 family under
distribution shift"* (2026-10-08, see `REPORT.md`), which positions itself
against arXiv:2610.00346 (Rafe & Das — 8 checkpoints / 6 families incl.
Laya; RQ2: per-task temperature fitting on held-out data). Our study is the
first calibration look at the Liquid d1 open-weight family (d1-omni-600M,
d1-3B), adds per-question-type (noul/choice/score) diagnosis, and documents
failure modes.

## Layout

```
decision_calib/
  models.py      # registry: load_model("d1-omni"|"d1-3b"|"opendecider-small"|"jeff-2b"|"jeff-0.8b") -> uniform interface
  datasets.py    # typed-decisions + OOD sets (ag_news, rotten_tomatoes, sst2)
  metrics.py     # accuracy, ECE (equal-mass), Brier, NLL, reliability data
  calibrate.py   # temperature scaling: fit/apply, global | per_type | per_task
scripts/
  build_bundle.py   # inlines the package into dist/measure_<model>.py (1 file for Colab)
  run_measure.py    # template executed on the Colab GPU VM -> predictions CSV
  run_rq2.py        # local: as-shipped metrics + temperature-fitting experiments
  make_figures.py   # local: reliability diagrams -> figures/
data/            # predictions_*.csv (full distributions), metrics_*.csv, temperatures.json
figures/         # rel_<model>_<dgroup>.png
dist/            # generated single-file Colab bundles (build artifact)
REPORT.md        # study report (Vietnamese, verdict-first)
```

## Reproduce the study

```bash
pip install -r requirements.txt
# 1. GPU inference (Colab T4; session already stopped, recreate if needed)
#    Per-model extras (only what you plan to measure):
#      d1-*:          (nothing extra; transformers>=5.15 + trust_remote_code)
#      opendecider:   pip uninstall -y torchao && pip install "opendecider[small]"
#      jeff-*:        auto-cloned from github.com/firelex/jeff by the loader
python scripts/build_bundle.py
colab new -s calib-d1 --gpu T4
colab install -s calib-d1 "transformers>=5.15" datasets pillow soundfile
colab exec -s calib-d1 -f dist/measure_d1-omni.py --timeout 1200
colab download -s calib-d1 /tmp/predictions_d1-omni.csv data/predictions_d1-omni.csv
colab exec -s calib-d1 -f dist/measure_d1-3b.py --timeout 1800
colab download -s calib-d1 /tmp/predictions_d1-3b.csv data/predictions_d1-3b.csv
colab stop -s calib-d1
# 2. Analysis (no GPU)
python scripts/run_rq2.py
python scripts/make_figures.py
```

## Add a new model (two steps, no analysis changes)

1. In `decision_calib/models.py`, write `load_<name>()` returning a handle,
   plus `predict_batch_<name>(handle, items, chunk)` returning raw per-question
   answer dicts, plus `normalize_<name>(ans, qtype) -> {"pred", "probs"}` if its
   answer format differs from the d1 family's.
2. Add one line to `REGISTRY`, e.g.:
   `"<name>": {"load": load_<name>, "predict_batch": predict_batch_<name>,
               "normalize": normalize_<name>, "chunk": 32},`

Then `python scripts/build_bundle.py` and run `dist/measure_<name>.py`.
`run_rq2.py` / `make_figures.py` pick up any `data/predictions_<name>.csv`
if you extend their model lists.

`laya` (convaiinnovations/laya-typed-decisions) is already registered as an
example of a non-d1 model (needs `pip install laya`, `USE_TF=0`); the d1
study cites its published numbers instead of rerunning it.

Cross-family extension (2026-10-08) added three measured checkpoints:
`opendecider-small` (`manjunathshiva/opendecider-small` — Qwen3-4B-Instruct-2507
+ LoRA; adapter merges LoRA manually layer-by-layer to avoid the 14.4GB peak
of `merge_and_unload()` on T4), `jeff-2b` / `jeff-0.8b`
(`mstrasser/Jeff-Qwen3.5-2B` / `mstrasser/Jeff-Qwen3.5-0.8B` — Qwen3.5 backbone
+ 255-option readout via the `firelex/jeff` package's `DecisionModel.predict`).
Bongard-mini (`AgentBull/bongard-mini`) was evaluated and skipped: 7.51B
encoder-decoder needs ~15GB BF16 for weights alone (per their README), which
does not fit a T4 15.36GB, and their `Predictor` API has no quantization path.

## Add a dataset

Write a loader in `decision_calib/datasets.py` returning cases
`{"domain","row_id","state","questions","gold"}` with questions in the
Decision Index schema and gold labels as strings, register it in `DATASETS`,
and reference it in `run_measure.py`'s loader list.

## Key finding (for the paper)

d1 models ship pre-calibrated via per-type temperatures in `config.json`
(as-shipped ECE 0.04–0.15), so held-out temperature fitting — the dramatic
fix in 2610.00346 (ECE 0.251 → 0.038) — yields only marginal gains here.
The per-type split reveals d1-omni's real weakness: overconfident
ordinal (score) answers (ECE 0.152, fitted T=2.35), fixed in d1-3B.
