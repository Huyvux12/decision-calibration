# CHANGELOG — packaging / QC pass (2026-10-08)

No new GPU measurements were run in this pass. All changes are to code,
docs, or packaging; experimental results are untouched.

## Fixed / improved

1. **`decision_calib/models.py` — lazy torch import.** The module-level
   `import torch` (only needed for the `dtype=torch.float16` default arg of
   `load_d1`) made the whole package unimportable on GPU-less machines.
   `load_d1` now takes `dtype=None` and imports torch inside the loader.
   `import decision_calib` now works with only numpy/scipy installed.
2. **`scripts/run_measure.py` — `--help`.** The template previously ignored
   `--help`; it now prints usage (`--probe-only`, `--out PATH`).
3. **`scripts/run_rq2.py`, `scripts/make_figures.py`,
   `scripts/build_bundle.py` — `--models` flag.** The 5-checkpoint model
   list was hardcoded in three places; it is now an argparse option with
   the same list as default, so a new model needs no script edits.
4. **`requirements.txt` — pinned + completed.** Lower-bound pins
   (verified on Colab 2026-10-08) plus previously missing dependencies:
   `huggingface_hub`, `safetensors`, and install notes for the
   model-specific extras (`opendecider[small]` incl. the Colab torchao
   conflict note, `laya`, `firelex/jeff` git clone).
5. **`decision_calib/models.py` docstring — opendecider package source.**
   The `opendecider` import had no documented install path; the loader
   docstring now records `pip install "opendecider[small]"`.
6. **`README.md` — per-model GPU install extras** added to the reproduce
   section.
7. **`REPORT.md` §10.2 — temperature mode labeled.** The table mixed
   per-type temperatures with global-mode ECE deltas without saying so;
   the section now states the ECE deltas use the **global** temperature
   (per-type/per-task values are in `data/metrics_rq2.csv`).
8. **`scripts/smoke_test.py` (new).** GPU-free test: package import +
   registry completeness, metrics on synthetic data (calibrated vs
   overconfident), temperature-fit recovery of a known T, full validation
   of all 5 predictions CSVs (schema, 2,200 rows, probs sum to 1, indices
   in range), metrics CSVs / temperatures.json parse, all 10 figures
   present. **All pass.**
9. **`dist/measure_*.py` rebuilt** from the updated template
   (`python scripts/build_bundle.py`; bundles compile).

## Verified (no change needed)

- All 5 `data/predictions_*.csv` share an identical schema, 2,200 rows
  each; every row's probabilities sum to 1 and indices are in range.
- REPORT.md tables spot-checked against `data/metrics_asshipped.csv`,
  `data/metrics_rq2.csv`, `data/temperatures.json`: §3, §4, §10.1, §10.2
  all match (rounding only). §10.2 ECE deltas are global-mode — now
  labeled (item 7).
- `python scripts/run_rq2.py` re-run on the frozen predictions is
  **byte-identical** (fixed seed 20261008; stratified splits deterministic).
- `python scripts/make_figures.py` regenerates all 10 reliability diagrams
  without errors.
- All 10 `sandbox://` figure links in REPORT.md resolve to existing files.
