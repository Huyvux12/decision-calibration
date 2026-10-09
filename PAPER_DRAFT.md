# When Does Shipped Calibration Transfer? A Case-Disjoint Study of Open Decision Models

**Working manuscript, 9 October 2026.** Authors and affiliations: to be supplied. Results combine a verified historical cross-family reanalysis and two completed, provenance-recorded Colab d1 runs on 3,000 balanced external items per checkpoint. Semantic OOD results for OpenDecider and Jeff remain limited to the historical probes.

## Abstract

Open decision models expose typed option probabilities without generating answer text. We investigate whether an additional temperature fitted on workflow decisions improves confidence quality and transfers to external tasks. A historical corpus covers five checkpoints across three release families; a new provenance-recorded extension evaluates both Liquid d1 checkpoints on 3,000 balanced external items, including five-level Yelp ratings. All workflow calibration splits and uncertainty estimates preserve the state as the sampling unit. For d1-omni, per-primitive scaling lowers mean workflow ECE from 0.0790 to 0.0458, yet raises Yelp ECE from 0.0418 to 0.1239 and log loss from 0.8108 to 0.8881 across twenty splits. Refitting-bootstrap intervals at the prespecified split support this deterioration. Its shipped temperature, by contrast, improves Yelp ECE relative to reconstructed raw probabilities from 0.1203 to 0.0418. The larger d1-3B does not exhibit the same consistent refit harm; its metric changes remain small or uncertain. Our findings document a task-specific counterexample to primitive-level calibration transfer and motivate evaluating probability quality on the destination task rather than treating source-task recalibration as a universal repair.

## 1. Introduction

Software agents and routing systems frequently need a choice among declared alternatives rather than a generated explanation. Decision models expose probabilities over those alternatives, making confidence thresholds operationally convenient. A probability is useful for such a threshold only to the extent that its interpretation survives changes in the request distribution, options, and target task.

We investigate three questions. **RQ1:** How does shipped probability quality differ across checkpoints and question primitives? **RQ2:** How stable are improvements from an additional temperature fitted on a small set of case-disjoint workflow states? **RQ3:** Do those improvements transfer to withheld workflows and external classification tasks?

Our contribution is an empirical extension covering the recently released open Liquid checkpoints alongside OpenDecider and Jeff, with a reproducible probability-level analysis. We audit evaluation units, preserve tied decisions under scaling, include refitting uncertainty, and separately report fidelity to soft teacher distributions. We add balanced semantic OOD evaluation of both Liquid checkpoints, including an ordinal task and a controlled inverse-temperature arm for d1-omni. We do not propose a new calibration algorithm or establish a causal effect of model family, architecture, or training pipeline.

## 2. Related work

Temperature scaling is a standard post-hoc procedure for probabilistic classifiers [1]. Rafe and Das study decision models and alternative classifiers under a shared decision-gate harness [2]. Their calibration evaluation already distinguishes shipped, raw, and recalibrated probabilities and includes case-aware uncertainty. Our implementation is an extension with different checkpoints and task samples, not an exact replication of that study.

The typed-decisions benchmark provides multiple typed questions per state and soft teacher-generated references [3]. Consequently, hard-label agreement measures agreement with that benchmark reference, while distributional fidelity measures agreement with its teacher probability spread. Neither is a substitute for independently assessed correctness. Model documentation supplies the interfaces and checkpoint identities for the three release families [4–7]. We avoid direct metric ranking against heterogeneous leaderboard reports.

## 3. Data and models

### 3.1 Checkpoints

We analyze LiquidAI/d1-omni-600M, LiquidAI/d1-3B, manjunathshiva/opendecider-small, mstrasser/Jeff-Qwen3.5-2B, and mstrasser/Jeff-Qwen3.5-0.8B. “Family” denotes release lineage here: Liquid, OpenDecider, and Jeff. This grouping does not imply identical architectures within a lineage. Predictions were supplied as CSV files from prior runs, reportedly using FP16 on a Colab T4. The historical files contain full option distributions rounded to six decimal places, but lack complete runtime manifests or immutable checkpoint hashes; we do not independently certify their hardware, timings, or historical revisions.

The new extension includes complete Colab runs for both Liquid checkpoints on the same exact inputs. Their resolved revisions are `02b55d7076f15129e59ab3f94783f32c4b088674` (omni) and `051bcc464b01b9f92942b364d9586b0ef5912432` (3B). Exported metadata records FP16 inference on Tesla T4 with PyTorch 2.11.0+cu130 and Transformers 5.19.0. We verify manifests, input hashes, row coverage, gold alignment, and exported source, rather than independently rerunning GPU inference. Each run contains 5,000 decisions with no recorded failures or missing outputs. Workflow selected options match the historical predictions; their probability differences are at float precision scale.

Source inspection of a current pinned d1-omni snapshot confirms explicit text temperatures indexed by primitive and option-count band. Inspection of the current d1-3B configuration and engine construction does not establish the same explicit calibration mechanism. We therefore do not attribute the two checkpoints' probability quality to a shared stored-temperature implementation.

### 3.2 Workflow data

The workflow benchmark contains four test workflows: agent-trace observability, customer service, invoice processing, and security incidents. There are 100 states per workflow and five questions per state, giving 400 states and 2,000 decisions: 600 binary (`noul`), 600 nominal (`choice`), and 800 ordinal (`score`). We treat the state as the independent sampling and splitting unit.

For historical predictions, we retrieved the benchmark's soft gold references, matched every state/question ID and hard label, and retained the retrieved records and source metadata. This hydration adds reference distributions; it does not independently recover the original historical input bytes.

### 3.3 Exploratory external tasks

Existing external predictions cover 40 Rotten Tomatoes reviews (20/class), 100 SST-2 validation examples (50/class), and 60 AG News items. AG News is skewed: 15 world, 11 sports, 5 business, and 29 science/technology examples. These 200 decisions include binary and nominal questions but no ordinal questions. They are small public benchmark probes, not evidence of universal distribution-shift robustness or training-data independence.

### 3.4 Balanced external extension

Both Liquid checkpoints are evaluated on the same 3,000 external items: AG News test 1,000 (250/class), Rotten Tomatoes test 500 (250/class), SST-2 validation 500 (250/class), and Yelp Review Full test 1,000 (200/star level). Selection uses seed 20261009 within label strata. We retain exact states, rubrics, references, original dataset row IDs, dataset hashes, and a shared input file SHA256. No selected states coincide after normalized exact-text comparison; this does not rule out training exposure or semantic near-duplicates.

Yelp's zero-based labels 0–4 are mapped to ordered Decision Index levels described as one through five stars. Only review text and the requested rubric enter the model. The target is the author's star rating inferred from review text, not an objective business-quality judgment. Workflow score questions comprise 700 decisions with four levels and 100 with five levels. Transfer to Yelp therefore changes both semantics/rubric and the level-count distribution; the experiment does not isolate their separate contributions.

## 4. Method

### 4.1 Probability validation and ties

Within each evaluation corpus, checkpoints cover identical unique `(domain,state,question)` keys: 2,200 per historical checkpoint and 5,000 per new d1 run. We validate finite nonnegative probabilities, near-unit total mass, valid option labels, and valid reference indices. We renormalize the tiny rounding discrepancy in total probability. Binary outputs exactly tied at 0.5 retain the originally reported selected option. A positive temperature preserves that selected argmax, including the tie convention; accuracy is therefore invariant under our scaling transformation.

### 4.2 Calibration and splits

For shipped probabilities p, we define an additional temperature t by

$$p_t(k)=\frac{p(k)^{1/t}}{\sum_j p(j)^{1/t}}.$$

We minimize hard-reference negative log likelihood over t in [0.05,10], using bounded optimization in log-temperature space. This reconstructs ordinary logit scaling up to probability rounding and numerical clipping. The fitted t is relative to the shipped distribution; it is not necessarily the model's total temperature relative to raw logits.

For each of twenty seeds (20261009 through 20261028), we select 20% of states within each workflow for fitting. All questions from a selected state remain together: 80 fit states/400 decisions and 320 evaluation states/1,600 decisions, with zero state overlap. Seeds characterize sensitivity to calibration-set selection, not independent replications of the benchmark.

We compare one global t, one t per primitive, and one t per workflow/primitive cell. Only global and per-primitive temperatures transfer to external tasks. A workflow-specific map has no entry for an unseen task; we exclude that condition rather than silently applying t=1. We also fit on three complete workflows and evaluate the fourth (leave-one-workflow-out), a complementary shift within the same benchmark.

### 4.3 Measures

Primary probability quality is hard-reference NLL; we also report multiclass Brier, top-label ECE, accuracy, and mean confidence. Multiclass Brier is the sum over classes, so its binary range is [0,2]. Top-label ECE compares predicted confidence with hard-reference agreement, using 15 equal-mass bins for continuity with the supplied analysis. Five- and ten-bin variants and ten-/fifteen-bin equal-width variants are retained as sensitivity checks. Equal-mass bins can split tied confidences; equal-width checks are particularly relevant in that setting.

For workflow soft references s, we separately compute KL(s||p), squared distance sum_k(p_k−s_k)^2, and s at the selected option (“soft accuracy”). For ordinal questions we report normalized ranked probability score: the mean squared difference between cumulative predicted and reference probabilities over the K−1 internal thresholds. Hard and soft reference versions are named separately in outputs.

### 4.4 Uncertainty

For the prespecified primary split, we generate 1,000 bootstrap replicates, resampling whole states separately within workflows/datasets in both calibration and evaluation sets. Each replicate refits its temperatures and compares scaled and shipped scores on the same evaluation resample. We report percentile intervals for after-minus-before differences. These intervals describe resampling uncertainty conditional on the fixed workflow structure and historical prediction corpus; they do not capture training randomness or unknown checkpoint changes. ECE is a nonsmooth, finite-sample-biased statistic. Intervals are exploratory estimation summaries, not multiplicity-corrected hypothesis tests.

## 5. Results

### 5.1 As-shipped performance

Table 1 reports all workflow decisions, without reserving a calibration subset. OOD columns refer only to the 200 historical items.

| Checkpoint | Workflow agreement | Workflow ECE15 | Workflow NLL | OOD agreement | OOD ECE15 |
|---|---:|---:|---:|---:|---:|
| d1-omni | 0.5770 | 0.0756 | 0.9434 | 0.8600 | 0.0712 |
| d1-3b | 0.6525 | 0.0494 | 0.8070 | 0.8900 | 0.0492 |
| opendecider-small | 0.6600 | 0.0697 | 0.8063 | 0.8800 | 0.0671 |
| jeff-2b | 0.5555 | 0.0941 | 0.9933 | 0.8100 | 0.0946 |
| jeff-0.8b | 0.4930 | 0.1374 | 1.2199 | 0.8250 | 0.0840 |

Pooling questions can conceal primitive-specific errors. Table 2 reports ECE separately. Score is the largest-ECE primitive for only two of the five checkpoints, not a cross-family universal failure mode.

| Checkpoint | Binary ECE15 | Choice ECE15 | Score ECE15 |
|---|---:|---:|---:|
| d1-omni | 0.0444 | 0.0759 | 0.1522 |
| d1-3b | 0.0497 | 0.0722 | 0.0710 |
| opendecider-small | 0.0774 | 0.0747 | 0.1140 |
| jeff-2b | 0.1049 | 0.1302 | 0.0778 |
| jeff-0.8b | 0.1286 | 0.1828 | 0.1411 |

### 5.2 Recalibration on withheld workflow states

Table 3 summarizes per-primitive scaling across twenty case-disjoint splits. Values are mean metrics across the same evaluation protocol, not mean bootstrap confidence limits.

| Checkpoint | Mean ECE15 shipped→scaled | Mean ΔNLL | Splits with lower ECE15 |
|---|---:|---:|---:|
| d1-omni | 0.0790→0.0458 | -0.0269 | 20/20 |
| d1-3b | 0.0510→0.0486 | 0.0020 | 14/20 |
| opendecider-small | 0.0707→0.0473 | -0.0289 | 19/20 |
| jeff-2b | 0.0957→0.0710 | -0.0171 | 20/20 |
| jeff-0.8b | 0.1388→0.0748 | -0.1338 | 20/20 |

The updated analysis changes the interpretation of d1-omni: further calibration is not uniformly negligible. Its per-type log-loss gain is stable across the examined splits. Jeff-0.8B has the largest average gain. In contrast, d1-3B's average NLL does not improve, even when ECE frequently decreases. These differences illustrate why a reduction in one calibration estimate should not be presented as a universal repair of probability quality.

Table 4 reports uncertainty in NLL differences at the prespecified split. A negative interval supports improvement in this estimation setting; intervals crossing zero indicate unresolved direction. This table does not establish causal differences between release families.

| Checkpoint | ID per-type ΔNLL | 95% cluster interval | OOD per-type ΔNLL | 95% cluster interval |
|---|---:|---|---:|---|
| d1-omni | -0.0268 | [-0.0410, -0.0031] | -0.0239 | [-0.0750, 0.0732] |
| d1-3b | 0.0045 | [-0.0033, 0.0226] | -0.0087 | [-0.0245, 0.0159] |
| opendecider-small | -0.0321 | [-0.0413, -0.0135] | 0.0368 | [0.0006, 0.1095] |
| jeff-2b | -0.0196 | [-0.0298, 0.0016] | -0.0480 | [-0.1056, 0.0432] |
| jeff-0.8b | -0.1321 | [-0.1694, -0.0888] | 0.0104 | [-0.1175, 0.1311] |

Most ECE-difference intervals include zero despite favorable point estimates; Jeff-0.8B is the clearest workflow ECE improvement. Full ECE intervals and alternative binning results are released rather than selecting the most favorable definition.

![Paired bootstrap differences](results_corrected/figures/calibration_delta_ci.png)

### 5.3 Historical small-probe transfer and metric disagreement

Global scaling improves average OOD NLL for d1-omni and Jeff-2B on these probes. Per-primitive scaling is less stable for d1-omni, because the benefit of a fitted workflow score temperature cannot directly help a set containing no score questions. OpenDecider has average per-type ID ΔNLL −0.0289 but OOD ΔNLL +0.0359; the primary-split OOD interval is positive, albeit close to zero at its lower endpoint. Its average OOD ECE changes little, underscoring a discrepancy between top-label calibration and the full probability distribution.

For Jeff-0.8B, global scaling improves mean OOD NLL by 0.0837 while increasing mean OOD ECE from 0.0840 to 0.0968. Per-type scaling increases mean OOD ECE further to 0.1338. These external ECE effects have broad bootstrap intervals. We describe the observed metric disagreement rather than claiming an established universal transfer failure.

### 5.4 Ordinal and teacher-reference diagnostics

The as-shipped soft-reference KL on all workflow questions is 0.3610 for d1-omni, 0.2891 for d1-3B, 0.2225 for OpenDecider, 0.3861 for Jeff-2B, and 0.5612 for Jeff-0.8B. On workflow score questions, soft-target RPS is 0.0735, 0.0436, 0.0354, 0.0714, and 0.1118 respectively. These are descriptions of fidelity to this teacher reference, not calibrated human-correctness probabilities or statistically established model rankings.

In leave-one-workflow-out score evaluation, d1-omni's ECE decreases in all four held workflows, but its score NLL worsens on security incidents. Jeff-2B's score NLL worsens in all four held workflows after per-type scaling. The results again caution against judging transfer by a pooled ECE alone. Complete workflow-specific tables are released.

### 5.5 Balanced external extension: completed d1 runs

Table 5 reports shipped probability quality on the new external tasks. The extended OOD macro mixture has pooled ECE15 0.0244 for omni and 0.0190 for 3B, but the task-specific results are more informative than those pooled values.

| Model | Task | n | Agreement/accuracy | ECE15 | NLL |
|---|---|---:|---:|---:|---:|
| d1-omni | AG News | 1000 | 0.8710 | 0.0213 | 0.3812 |
| d1-omni | Rotten Tomatoes | 500 | 0.8760 | 0.0651 | 0.3412 |
| d1-omni | SST-2 | 500 | 0.9260 | 0.0350 | 0.2363 |
| d1-omni | Yelp rating | 1000 | 0.6340 | 0.0418 | 0.8108 |
| d1-3B | AG News | 1000 | 0.9030 | 0.0255 | 0.3199 |
| d1-3B | Rotten Tomatoes | 500 | 0.8880 | 0.0339 | 0.2736 |
| d1-3B | SST-2 | 500 | 0.9520 | 0.0446 | 0.1712 |
| d1-3B | Yelp rating | 1000 | 0.6310 | 0.0632 | 0.8264 |

The two checkpoints' Yelp accuracies are nearly equal: 0.634 versus 0.631. A paired bootstrap interval for 3B-minus-omni accuracy is [−0.0310,0.0210]. Their ECE and NLL comparison intervals also include zero; we do not claim an established accuracy or calibration ranking from those point estimates.

Workflow-fitted per-type scaling harms omni on Yelp in all twenty examined splits. Table 6 reports mean scores; the fitted temperature is an additional scaling relative to shipped probabilities.

| Checkpoint | Yelp ECE15 shipped→per-type | Yelp NLL shipped→per-type | Splits with higher NLL |
|---|---:|---:|---:|
| d1-omni | 0.0418→0.1239 | 0.8108→0.8881 | 20/20 |
| d1-3B | 0.0632→0.0550 | 0.8264→0.8269 | 10/20 |

At the prespecified split, omni's per-type Yelp ECE delta is +0.0737 with a 95% refitting-bootstrap interval [0.0239,0.1218]; its NLL delta is +0.0678 [0.0278,0.1238]. For 3B, the corresponding ECE delta is −0.0363 [−0.0456,0.0165], while NLL delta is +0.0035 [−0.0069,0.0322]. Thus the strong ordinal transfer deterioration is checkpoint-specific rather than a shared property of the two Liquid models.

Taskwise results explain why pooled or small-sample transfer conclusions can change. Omni's global temperature improves sentiment log loss in all twenty splits but increases Yelp and news log loss in all twenty; per-type scaling raises pooled extended OOD NLL by 0.0293 on average. The extended per-type OOD NLL interval at the primary split is [0.0049,0.0586]. Historical small-probe conclusions should not be generalized to this broader task mixture.

### 5.6 Explicit shipped-temperature ablation

For the new omni snapshot, we reconstruct raw text probabilities by raising each shipped probability to its documented temperature and renormalizing. Every stored raw row matches that transformation; selected options remain unchanged. This is a direct ablation of the explicit scaling operation on the same input predictions, not a causal investigation of training.

On Yelp, raw ECE15 is 0.1203, shipped ECE15 is 0.0418, and primary-split ID-refitted ECE15 is 0.1155. Raw NLL is 0.8759, shipped NLL 0.8108, and primary-split refitted NLL 0.8786. The shipped-minus-raw Yelp ECE difference has a paired bootstrap interval [−0.0987,−0.0334], and the NLL difference [−0.0960,−0.0346]. The stored temperature benefits the destination task, while fitting an additional workflow temperature can undo that benefit. No raw arm is asserted for 3B because its exported configuration does not establish the same explicit mechanism.

![Taskwise temperature effects](colab_complete/taskwise_temperature_effects.png)

## 6. Discussion

Shipped confidence must be interpreted relative to a task and readout. Additional scaling can help a checkpoint with substantial workflow overconfidence while providing little NLL benefit for an already strong checkpoint. More specialized calibration may reduce bias within a workflow but increase variance or transfer poorly when the semantics of the new task differ.

The current results are consistent with this task dependence; they do not identify the training-pipeline component responsible. Our completed raw-versus-shipped arm directly measures the explicit stored temperature for omni. It confirms an effect of that scaling operation on these tasks; it does not attribute cross-family differences to architecture, distillation, or training calibration.

For deployed decision gates, calibration estimation should precede operational threshold selection on separately held data, and threshold risk must be evaluated on the intended deployment distribution. This study does not certify any threshold or guarantee a target error rate under shift.

## 7. Limitations

Historical predictions lack immutable runtime provenance and are rounded. Workflow gold is synthetic and teacher-derived; state-level clustering corrects dependence among questions but does not convert the benchmark into human ground truth. Historical OOD items are few, mostly sentiment, and class-skewed for news. The balanced extension repairs those sampling limitations for the two d1 checkpoints; it does not extend OpenDecider or Jeff to the new ordinal task. Public benchmarks may have appeared in training. We do not infer a parameter-scaling law from the two Jeff checkpoints. Binning, sample size, and the primary split affect ECE; bootstrap intervals are not immune to ECE bias. Uniform predictions can obtain deceptively small calibration errors without meaningful discrimination. The new two-d1 runs provide pinned inference provenance, but the other checkpoints remain historical observations. The ordinal experiment changes semantic task and option-count mix simultaneously; it does not isolate the source of transfer failure.

## 8. Reproducibility and completion protocol

The release contains the historical distributions, hydrated references, input-key audits, twenty case splits, primary temperature maps, metric CSVs, refitting-bootstrap results, figures, and an executable standalone Colab notebook. Original claims superseded by this audit are retained only as historical material.

The notebook's default extension is now complete: both d1 checkpoints have 2,000 workflow and 3,000 balanced external predictions. The complete export contains exact cases, checkpoint/dataset hashes, package versions, error logs, and coverage manifests. Added CPU scripts produce taskwise raw/shipped and refitted analyses, refitting-bootstrap intervals, and paired checkpoint comparisons. Newly generated tables distinguish twenty-split means from primary-split point estimates. Any further semantic OOD cross-family extension must use the same case hash; no such new OpenDecider/Jeff experiment is represented here.

## 9. Conclusion

Case-disjoint recalibration improves workflow probability quality for several checkpoints, yet this benefit does not automatically survive a task change. The completed balanced extension provides a specific ordinal counterexample: omni's source-fitted per-type temperature improves workflow predictions but consistently worsens Yelp probability quality, while its explicit shipped temperature improves that destination task relative to raw probabilities. The larger d1 checkpoint does not reproduce the same strong harm. Calibration reports should preserve the state as their evaluation unit, distinguish target references, quantify refitting uncertainty, and test the destination task rather than relying on a primitive label alone.

## References

[1] Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. 2017. *On Calibration of Modern Neural Networks.* ICML, PMLR 70. https://proceedings.mlr.press/v70/guo17a.html

[2] Rafe, A., and Das, S. 2026. *Benchmarking System One decision models against trained classifiers and language models for automated decision gates.* arXiv:2610.00346v2. https://arxiv.org/html/2610.00346v2

[3] LocalLLaMA. *Typed Decisions dataset.* https://huggingface.co/datasets/LocalLLaMA/typed-decisions

[4] Liquid AI. *d1-omni-600M model card.* https://huggingface.co/LiquidAI/d1-omni-600M

[5] Liquid AI. *d1-3B model card.* https://huggingface.co/LiquidAI/d1-3B

[6] manjunathshiva. *OpenDecider-small model card.* https://huggingface.co/manjunathshiva/opendecider-small

[7] mstrasser. *Jeff-Qwen3.5 checkpoints.* https://huggingface.co/mstrasser/Jeff-Qwen3.5-2B and https://huggingface.co/mstrasser/Jeff-Qwen3.5-0.8B

[8] Yelp. *Yelp Review Full dataset.* https://huggingface.co/datasets/Yelp/yelp_review_full
