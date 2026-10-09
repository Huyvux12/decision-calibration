# Verified calibration results

Delta = after − before. Negative losses/ECE indicate improvement. Repeated splits are dependent stability measurements; their spread is not a confidence interval. Bootstrap intervals resample cases and refit temperatures.

## As shipped: whole datasets

| Model | Dataset | n | Accuracy | ECE mass15 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| d1-omni | in-domain | 2000 | 0.5770 | 0.0756 | 0.9434 | 0.5403 |
| d1-omni | ood | 3000 | 0.8020 | 0.0244 | 0.4936 | 0.2752 |
| d1-3b | in-domain | 2000 | 0.6525 | 0.0494 | 0.8070 | 0.4591 |
| d1-3b | ood | 3000 | 0.8180 | 0.0190 | 0.4563 | 0.2580 |

## Repeated case-disjoint calibration splits (all types pooled)

| Model | Eval | Mode | Splits | Mean ECE before→after | Mean ΔNLL | ECE improves in |
|---|---|---|---:|---:|---:|---:|
| d1-3b | in-domain | global | 20 | 0.0510→0.0466 | 0.0008 | 18/20 |
| d1-3b | in-domain | per_task | 20 | 0.0510→0.0529 | 0.0005 | 9/20 |
| d1-3b | in-domain | per_type | 20 | 0.0510→0.0487 | 0.0020 | 14/20 |
| d1-3b | ood | global | 20 | 0.0190→0.0208 | 0.0028 | 14/20 |
| d1-3b | ood | per_type | 20 | 0.0190→0.0229 | 0.0047 | 5/20 |
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
| d1-3b | in-domain | global | nll | -0.0001 | [-0.0050, 0.0115] |
| d1-3b | in-domain | global | ece_mass_15 | -0.0022 | [-0.0190, 0.0175] |
| d1-3b | ood | global | nll | 0.0031 | [-0.0011, 0.0166] |
| d1-3b | ood | global | ece_mass_15 | -0.0005 | [-0.0090, 0.0255] |
| d1-3b | in-domain | per_type | nll | 0.0045 | [-0.0033, 0.0226] |
| d1-3b | in-domain | per_type | ece_mass_15 | -0.0034 | [-0.0191, 0.0259] |
| d1-3b | ood | per_type | nll | 0.0010 | [-0.0027, 0.0155] |
| d1-3b | ood | per_type | ece_mass_15 | 0.0014 | [-0.0127, 0.0218] |

## Leave-one-workflow-out (per-type, score questions only)

| Model | Held workflow | ECE before→after | NLL before→after |
|---|---|---:|---:|
| d1-omni | typed_agent_trace_observability | 0.1561→0.1028 | 1.2013→1.1145 |
| d1-omni | typed_customer_service | 0.2336→0.1234 | 1.3461→1.1593 |
| d1-omni | typed_invoice_processing | 0.1779→0.1118 | 1.3613→1.3011 |
| d1-omni | typed_security_incidents | 0.1253→0.0901 | 1.1670→1.2385 |
| d1-3b | typed_agent_trace_observability | 0.1501→0.1553 | 1.0287→1.0285 |
| d1-3b | typed_customer_service | 0.0693→0.0702 | 0.7762→0.7837 |
| d1-3b | typed_invoice_processing | 0.1669→0.1833 | 0.9740→1.0414 |
| d1-3b | typed_security_incidents | 0.1236→0.1242 | 0.9918→1.0190 |
