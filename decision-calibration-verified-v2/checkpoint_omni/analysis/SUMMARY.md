# Verified calibration results

Delta = after − before. Negative losses/ECE indicate improvement. Repeated splits are dependent stability measurements; their spread is not a confidence interval. Bootstrap intervals resample cases and refit temperatures.

## As shipped: whole datasets

| Model | Dataset | n | Accuracy | ECE mass15 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| d1-omni | in-domain | 2000 | 0.5770 | 0.0756 | 0.9434 | 0.5403 |
| d1-omni | ood | 3000 | 0.8020 | 0.0244 | 0.4936 | 0.2752 |

## Repeated case-disjoint calibration splits (all types pooled)

| Model | Eval | Mode | Splits | Mean ECE before→after | Mean ΔNLL | ECE improves in |
|---|---|---|---:|---:|---:|---:|
| d1-omni | in-domain | global | 20 | 0.0790→0.0479 | -0.0170 | 20/20 |
| d1-omni | in-domain | per_task | 20 | 0.0790→0.0523 | -0.0221 | 20/20 |
| d1-omni | in-domain | per_type | 20 | 0.0790→0.0458 | -0.0269 | 20/20 |
| d1-omni | ood | global | 20 | 0.0244→0.0355 | 0.0045 | 4/20 |
| d1-omni | ood | per_type | 20 | 0.0244→0.0558 | 0.0293 | 0/20 |

## Paired cluster bootstrap (seed 0; fit refitted each replicate)

| Model | Eval | Mode | Metric | Δ | 95% percentile interval |
|---|---|---|---|---:|---|
| d1-omni | in-domain | global | nll | -0.0164 | [-0.0264, -0.0035] |
| d1-omni | in-domain | global | ece_mass_15 | -0.0349 | [-0.0471, 0.0033] |
| d1-omni | ood | global | nll | 0.0100 | [-0.0033, 0.0348] |
| d1-omni | ood | global | ece_mass_15 | 0.0223 | [-0.0056, 0.0572] |
| d1-omni | in-domain | per_type | nll | -0.0268 | [-0.0410, -0.0031] |
| d1-omni | in-domain | per_type | ece_mass_15 | -0.0344 | [-0.0481, 0.0044] |
| d1-omni | ood | per_type | nll | 0.0199 | [0.0049, 0.0586] |
| d1-omni | ood | per_type | ece_mass_15 | 0.0268 | [0.0041, 0.0580] |

## Leave-one-workflow-out (per-type, score questions only)

| Model | Held workflow | ECE before→after | NLL before→after |
|---|---|---:|---:|
| d1-omni | typed_agent_trace_observability | 0.1561→0.1028 | 1.2013→1.1145 |
| d1-omni | typed_customer_service | 0.2336→0.1234 | 1.3461→1.1593 |
| d1-omni | typed_invoice_processing | 0.1779→0.1118 | 1.3613→1.3011 |
| d1-omni | typed_security_incidents | 0.1253→0.0901 | 1.1670→1.2385 |
