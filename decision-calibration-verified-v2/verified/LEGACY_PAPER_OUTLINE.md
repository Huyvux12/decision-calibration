# PAPER OUTLINE — Calibration of zero-token decision models under distribution shift

Paper-facing scaffold (English). Concrete numbers below come from
`REPORT.md` / `data/*.csv` (2026-10-08, 5 checkpoints, Colab T4).

## Proposed title

**"Calibration Is a Pipeline Property: A Cross-Family Study of Zero-Token
Decision Models under Distribution Shift"**

Alternates:
- "Ships Calibrated? Measuring and Repairing Calibration in Open-Weight
  Decision Models"
- "Beyond Accuracy: Per-Type Calibration Diagnosis of System-One Decision
  Models"

## Abstract skeleton (~150 words, numbers filled)

> Zero-token decision models return probabilities over fixed options with
> no text generation, and vendors increasingly market them as
> "calibrated". We measure calibration of **5 open-weight checkpoints
> from 4 families** (Liquid d1-omni-600M, d1-3B; OpenDecider-small;
> Jeff-Qwen3.5-2B/0.8B) on 2,000 in-domain decisions (typed-decisions test
> split) and 200 out-of-domain decisions (sentiment, SST-2, AG News),
> reporting accuracy, ECE, Brier and NLL per question type
> (yes/no, choice, ordinal score). Finding: **"ships calibrated" is not a
> property of the paradigm but of the training pipeline.** Liquid d1 ships
> well-calibrated (as-shipped ECE 0.04–0.15; held-out temperature refit
> barely helps, 0.062→0.051). OpenDecider-small ships **under**confident
> (fitted T 0.59/0.69 on score/choice; refit halves ECE 0.070→0.037).
> Jeff ships **over**confident, worse at smaller scale (0.8B needs T≈4.0;
> OOD transfer fails, 0.084→0.112). Ordinal score questions are the
> hardest to calibrate across all families. We release code, predictions
> with full distributions, and a reusable harness.

## Section-by-section outline → evidence mapping

1. **Introduction**
   - Decision models vs LLM-as-judge: zero output tokens, thresholdable
     probabilities for automated gates.
   - Vendor calibration claims are largely unverified (Liquid advertises
     "calibrated probabilities" with no checkable numbers).
   - Contributions: (a) first calibration study of the Liquid d1
     open-weight family; (b) cross-family comparison (5 checkpoints /
     4 families); (c) per-question-type diagnosis; (d) RQ2-style
     temperature-refit replication; (e) reusable harness + full
     probability distributions released.
2. **Related Work**
   - Rafe & Das, arXiv:2610.00346 (Sept/Oct 2026): 8 checkpoints /
     6 families incl. Laya; RQ2 per-task temperature fitting on held-out
     data (ECE 0.251→0.038). **Positioning:** their families exclude the
     d1 open weights (released 07/10/2026); they diagnose at task level,
     we diagnose per question type; we replicate their RQ2 protocol and
     show where its dramatic effect does / does not transfer.
   - Calibration literature: Guo et al. 2017 (temperature scaling);
     proper scoring rules (Brier decomposition); RLCD-style training
     with proper scores (laya model card).
   - Decision-model landscape: TypeSafe Jev, Laya (typed-decisions
     leaderboard), OpenDecider, Jeff, Decider.
3. **Method**
   - Models: table of 5 checkpoints (params, backbone, HF repo) →
     REPORT §10 table + `decision_calib/models.py` registry.
   - Datasets: typed-decisions test (400 cases / 2,000 decisions,
     4 workflows, noul 600 / choice 600 / score 800) + OOD (Rotten
     Tomatoes balanced 40, SST-2 val balanced 100, AG News 60, flagged
     skew) → REPORT §2, `decision_calib/datasets.py`.
   - Metrics: accuracy, mean confidence, ECE (equal-mass, 15 bins),
     multiclass Brier (note the [0,2] scale for binary noul),
     NLL → `decision_calib/metrics.py`.
   - RQ2 protocol: fit T on 20% in-domain held-out (stratified,
     seed 20261008) and 50% OOD; modes global / per-type / per-task;
     eval on remainder + OOD transfer → REPORT §2/§4,
     `decision_calib/calibrate.py`, `scripts/run_rq2.py`.
4. **Experiments / Results**
   - 4.1 As-shipped calibration (REPORT §3 table, §10.1 table;
     `data/metrics_asshipped.csv`).
   - 4.2 Per-type diagnosis: d1-omni score overconfidence (ECE 0.152,
     T=2.35) fixed in d1-3B; score hardest across families
     (REPORT §5, §10.3; figures `rel_*_indomain.png`).
   - 4.3 RQ2 replication: refit helps dramatically only where the
     model ships miscalibrated (OpenDecider 0.070→0.037); d1 barely
     moves; Jeff-0.8B OOD transfer fails (REPORT §4 table, §10.2 table;
     `data/metrics_rq2.csv`, `data/temperatures.json`).
   - 4.4 Reliability diagrams (`figures/rel_*.png`, 10 total).
5. **Discussion**
   - Calibration is a pipeline property, not a paradigm property:
     per-type temperatures shipped in config (d1) vs distillation
     from calibrated teachers (OpenDecider) vs none (Jeff).
   - Practical rule: never trust as-shipped confidence of an unfamiliar
     decision model — fit temperature on a small slice of your own
     task data (cheap, no retraining).
   - Score/ordinal questions deserve dedicated calibration attention.
6. **Limitations** (REPORT §7): OOD n=200, no score-type OOD items;
   AG News skew; 2610.00346 / laya numbers cited not rerun; fp16 run
   noise ±0.002; temperature on log-probs not logits.
7. **Conclusion** — one paragraph restating the headline + the practical
   rule + released artifacts.

## To-do before submission

- [ ] Authors / affiliations / corresponding author (placeholders).
- [ ] Venue decision (workshop vs conference vs journal; check
      arXiv:2610.00346's eventual venue to avoid direct collision).
- [ ] Strengthening experiments (pick at least one):
  - [ ] One more family on a bigger GPU (Bongard-mini needs ~15GB+;
        currently skipped — documented in REPORT §10).
  - [ ] Score-type OOD set (current OOD has no ordinal questions).
  - [ ] Confidence intervals / significance for the ECE deltas
        (bootstrap over the 2,200 rows; currently point estimates).
  - [ ] Human-rater validation on a small subset (gold-label noise check).
- [ ] Release: GitHub repo + Zenodo DOI for the bundle; predictions CSVs
      as an HF dataset for reusability.
- [ ] Ethics / broader impact paragraph (decision gates in production:
      miscalibrated confidence → wrong auto-escalation).
- [ ] Proofread numbers once more against `data/*.csv` after any change
      (CHANGELOG documents the 2026-10-08 QC pass).
