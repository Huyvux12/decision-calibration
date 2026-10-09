# Verified calibration results

Delta = after − before. Negative losses/ECE indicate improvement. Repeated splits are dependent stability measurements; their spread is not a confidence interval. Bootstrap intervals resample cases and refit temperatures.

## As shipped: whole datasets

| Model | Dataset | n | Accuracy | ECE mass15 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|
| d1-omni | in-domain | 2000 | 0.5770 | 0.0756 | 0.9434 | 0.5403 |
| d1-omni | ood | 200 | 0.8600 | 0.0712 | 0.4473 | 0.2265 |
| d1-3b | in-domain | 2000 | 0.6525 | 0.0494 | 0.8070 | 0.4591 |
| d1-3b | ood | 200 | 0.8900 | 0.0492 | 0.3060 | 0.1689 |
| opendecider-small | in-domain | 2000 | 0.6600 | 0.0697 | 0.8063 | 0.4634 |
| opendecider-small | ood | 200 | 0.8800 | 0.0671 | 0.3275 | 0.1819 |
| jeff-2b | in-domain | 2000 | 0.5555 | 0.0941 | 0.9933 | 0.5716 |
| jeff-2b | ood | 200 | 0.8100 | 0.0946 | 0.5097 | 0.2735 |
| jeff-0.8b | in-domain | 2000 | 0.4930 | 0.1374 | 1.2199 | 0.6642 |
| jeff-0.8b | ood | 200 | 0.8250 | 0.0840 | 0.5879 | 0.2895 |

## Repeated case-disjoint calibration splits (all types pooled)

| Model | Eval | Mode | Splits | Mean ECE before→after | Mean ΔNLL | ECE improves in |
|---|---|---|---:|---:|---:|---:|
| d1-3b | in-domain | global | 20 | 0.0510→0.0466 | 0.0008 | 18/20 |
| d1-3b | in-domain | per_task | 20 | 0.0510→0.0529 | 0.0005 | 9/20 |
| d1-3b | in-domain | per_type | 20 | 0.0510→0.0486 | 0.0020 | 14/20 |
| d1-3b | ood | global | 20 | 0.0492→0.0477 | 0.0008 | 15/20 |
| d1-3b | ood | per_type | 20 | 0.0492→0.0493 | -0.0002 | 13/20 |
| d1-omni | in-domain | global | 20 | 0.0790→0.0479 | -0.0170 | 20/20 |
| d1-omni | in-domain | per_task | 20 | 0.0790→0.0523 | -0.0221 | 20/20 |
| d1-omni | in-domain | per_type | 20 | 0.0790→0.0458 | -0.0269 | 20/20 |
| d1-omni | ood | global | 20 | 0.0712→0.0563 | -0.0387 | 20/20 |
| d1-omni | ood | per_type | 20 | 0.0712→0.0725 | 0.0067 | 7/20 |
| jeff-0.8b | in-domain | global | 20 | 0.1388→0.0626 | -0.1273 | 20/20 |
| jeff-0.8b | in-domain | per_task | 20 | 0.1388→0.0687 | -0.1561 | 20/20 |
| jeff-0.8b | in-domain | per_type | 20 | 0.1388→0.0748 | -0.1338 | 20/20 |
| jeff-0.8b | ood | global | 20 | 0.0840→0.0968 | -0.0837 | 8/20 |
| jeff-0.8b | ood | per_type | 20 | 0.0840→0.1338 | -0.0188 | 3/20 |
| jeff-2b | in-domain | global | 20 | 0.0957→0.0804 | -0.0140 | 20/20 |
| jeff-2b | in-domain | per_task | 20 | 0.0957→0.0583 | -0.0393 | 20/20 |
| jeff-2b | in-domain | per_type | 20 | 0.0957→0.0710 | -0.0171 | 20/20 |
| jeff-2b | ood | global | 20 | 0.0946→0.0865 | -0.0468 | 17/20 |
| jeff-2b | ood | per_type | 20 | 0.0946→0.0745 | -0.0467 | 20/20 |
| opendecider-small | in-domain | global | 20 | 0.0707→0.0483 | -0.0242 | 19/20 |
| opendecider-small | in-domain | per_task | 20 | 0.0707→0.0482 | -0.0274 | 19/20 |
| opendecider-small | in-domain | per_type | 20 | 0.0707→0.0473 | -0.0289 | 19/20 |
| opendecider-small | ood | global | 20 | 0.0671→0.0735 | 0.0214 | 2/20 |
| opendecider-small | ood | per_type | 20 | 0.0671→0.0657 | 0.0359 | 7/20 |

## Paired cluster bootstrap (seed 0; fit refitted each replicate)

| Model | Eval | Mode | Metric | Δ | 95% percentile interval |
|---|---|---|---|---:|---|
| d1-omni | in-domain | global | nll | -0.0164 | [-0.0264, -0.0035] |
| d1-omni | in-domain | global | ece_mass_15 | -0.0349 | [-0.0471, 0.0033] |
| d1-omni | ood | global | nll | -0.0456 | [-0.0944, -0.0019] |
| d1-omni | ood | global | ece_mass_15 | -0.0223 | [-0.0346, 0.0294] |
| d1-omni | in-domain | per_type | nll | -0.0268 | [-0.0410, -0.0031] |
| d1-omni | in-domain | per_type | ece_mass_15 | -0.0344 | [-0.0481, 0.0044] |
| d1-omni | ood | per_type | nll | -0.0239 | [-0.0750, 0.0732] |
| d1-omni | ood | per_type | ece_mass_15 | -0.0162 | [-0.0313, 0.0248] |
| d1-3b | in-domain | global | nll | -0.0001 | [-0.0050, 0.0115] |
| d1-3b | in-domain | global | ece_mass_15 | -0.0022 | [-0.0193, 0.0174] |
| d1-3b | ood | global | nll | 0.0004 | [-0.0135, 0.0189] |
| d1-3b | ood | global | ece_mass_15 | -0.0024 | [-0.0203, 0.0257] |
| d1-3b | in-domain | per_type | nll | 0.0045 | [-0.0033, 0.0226] |
| d1-3b | in-domain | per_type | ece_mass_15 | -0.0034 | [-0.0192, 0.0257] |
| d1-3b | ood | per_type | nll | -0.0087 | [-0.0245, 0.0159] |
| d1-3b | ood | per_type | ece_mass_15 | -0.0082 | [-0.0299, 0.0230] |
| opendecider-small | in-domain | global | nll | -0.0253 | [-0.0344, -0.0114] |
| opendecider-small | in-domain | global | ece_mass_15 | -0.0235 | [-0.0433, 0.0077] |
| opendecider-small | ood | global | nll | 0.0225 | [-0.0113, 0.0732] |
| opendecider-small | ood | global | ece_mass_15 | 0.0080 | [-0.0348, 0.0261] |
| opendecider-small | in-domain | per_type | nll | -0.0321 | [-0.0413, -0.0135] |
| opendecider-small | in-domain | per_type | ece_mass_15 | -0.0342 | [-0.0484, 0.0087] |
| opendecider-small | ood | per_type | nll | 0.0368 | [0.0006, 0.1095] |
| opendecider-small | ood | per_type | ece_mass_15 | -0.0182 | [-0.0322, 0.0563] |
| jeff-2b | in-domain | global | nll | -0.0151 | [-0.0257, 0.0004] |
| jeff-2b | in-domain | global | ece_mass_15 | -0.0195 | [-0.0393, 0.0009] |
| jeff-2b | ood | global | nll | -0.0543 | [-0.1024, -0.0106] |
| jeff-2b | ood | global | ece_mass_15 | -0.0101 | [-0.0343, 0.0278] |
| jeff-2b | in-domain | per_type | nll | -0.0196 | [-0.0298, 0.0016] |
| jeff-2b | in-domain | per_type | ece_mass_15 | -0.0226 | [-0.0503, 0.0038] |
| jeff-2b | ood | per_type | nll | -0.0480 | [-0.1056, 0.0432] |
| jeff-2b | ood | per_type | ece_mass_15 | -0.0191 | [-0.0432, 0.0694] |
| jeff-0.8b | in-domain | global | nll | -0.1250 | [-0.1605, -0.0837] |
| jeff-0.8b | in-domain | global | ece_mass_15 | -0.0750 | [-0.1048, -0.0314] |
| jeff-0.8b | ood | global | nll | -0.0797 | [-0.1867, 0.0339] |
| jeff-0.8b | ood | global | ece_mass_15 | 0.0234 | [-0.0367, 0.1029] |
| jeff-0.8b | in-domain | per_type | nll | -0.1321 | [-0.1694, -0.0888] |
| jeff-0.8b | in-domain | per_type | ece_mass_15 | -0.0720 | [-0.1031, -0.0203] |
| jeff-0.8b | ood | per_type | nll | 0.0104 | [-0.1175, 0.1311] |
| jeff-0.8b | ood | per_type | ece_mass_15 | 0.0792 | [-0.0141, 0.1643] |

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
| opendecider-small | typed_agent_trace_observability | 0.1688→0.1598 | 1.1532→1.2095 |
| opendecider-small | typed_customer_service | 0.1270→0.0788 | 0.7990→0.7274 |
| opendecider-small | typed_invoice_processing | 0.1252→0.1182 | 0.9023→0.8244 |
| opendecider-small | typed_security_incidents | 0.1604→0.0812 | 0.9185→0.8212 |
| jeff-2b | typed_agent_trace_observability | 0.1078→0.1033 | 1.2365→1.2440 |
| jeff-2b | typed_customer_service | 0.1678→0.1898 | 0.9968→1.0109 |
| jeff-2b | typed_invoice_processing | 0.2203→0.2735 | 1.2530→1.3813 |
| jeff-2b | typed_security_incidents | 0.1662→0.1522 | 1.2436→1.2789 |
| jeff-0.8b | typed_agent_trace_observability | 0.1667→0.0821 | 1.3852→1.3077 |
| jeff-0.8b | typed_customer_service | 0.0892→0.2110 | 1.1553→1.2364 |
| jeff-0.8b | typed_invoice_processing | 0.3100→0.2476 | 1.6876→1.3392 |
| jeff-0.8b | typed_security_incidents | 0.2416→0.1716 | 1.6919→1.5392 |
