# Báo cáo nghiên cứu hoàn chỉnh và bàn giao minh bạch

**Ngày: 09/10/2026.** Chủ đề: *When Does Shipped Calibration Transfer? A Case-Disjoint Study of Open Decision Models*.

## 1. Kết luận hiện tại

Đã đủ thực nghiệm để hoàn thiện một paper với phạm vi: tái phân tích corpus lịch sử của năm checkpoint và extension OOD cân bằng cho hai Liquid d1. Kết quả nổi bật là một counterexample cụ thể: temperature fit từ workflow có thể cải thiện workflow nhưng làm xấu xác suất trên task ordinal mới cùng primitive. Không cần thêm inference để viết xong phần hai d1; các run mới cho OpenDecider/Jeff chỉ cần nếu mở rộng kết luận semantic OOD sang nhiều họ model.

- d1-omni: per-type refit giảm workflow ECE trung bình 20 splits từ 0.0790 xuống 0.0458. Trên Yelp, ECE tăng từ 0.0418 lên 0.1239, NLL tăng từ 0.8108 lên 0.8881; NLL xấu hơn ở 20/20 splits.
- Bootstrap tại split chính có refit temperature: omni Yelp ΔNLL +0.0678, khoảng percentile 95% [0.0278, 0.1238]; ΔECE +0.0737, khoảng [0.0239, 0.1218]. Đây là point estimates tại một split, khác trung bình 20 splits.
- Shipped temperature của omni cải thiện Yelp so với raw reconstructed: ECE 0.1203→0.0418. Điều này cho thấy tác động của phép scaling phụ thuộc temperature và task, không phải mọi calibration đều gây hại.
- d1-3B không có cùng tác động NLL nhất quán. Trên Yelp, per-type refit giảm ECE trung bình nhưng NLL xấu hơn ở 10/20 splits; khoảng bootstrap của ΔNLL chứa 0.

Đóng góp là empirical evaluation và protocol có thể tái hiện, không phải một thuật toán calibration mới. Temperature scaling và tính phụ thuộc task đã có trong prior work. Không tuyên bố nghiên cứu đầu tiên, quy luật mọi model, hoặc nguyên nhân do kiến trúc/training. Khả năng được chấp nhận cần đánh giá theo venue và sự khác biệt với prior work; số liệu hiện tại không tự xác nhận novelty.

## 2. Dữ liệu thực sự có trong gói

| Corpus | Checkpoints | Decisions/model | Input và provenance | Vai trò |
|---|---|---:|---|---|
| Lịch sử | d1-omni, d1-3B, OpenDecider-small, Jeff-2B, Jeff-0.8B | 2.200 | CSV 6 chữ số, hydrated teacher gold; runtime/historical revision không đầy đủ | Exploratory context, 5 checkpoints/3 release lineages |
| Colab mới | d1-omni, d1-3B | 5.000 | Exact cases, immutable model revisions, manifest, source, environment, logs | Kết quả chính trên OOD cân bằng |

Tổng cộng 11.000 distributions lịch sử và 10.000 distributions mới. Đây không phải 21.000 observations độc lập: hai corpus lặp workflow, nhiều model dùng chung cases, và mỗi workflow case có năm câu hỏi. Mỗi run mới gồm 400 workflow cases × 5 decisions và 3.000 external cases × 1 decision, tổng 3.400 cases. Hai model mới dùng chung exact inputs.

Historical OOD chỉ có 200 items: Rotten 40, SST-2 100, AG News 60; AG News có nhãn 15/11/5/29. New OOD có AG News 1.000 (250/class), Rotten 500 (250/class), SST-2 500 (250/class), Yelp 1.000 (200/star). Không gộp hai corpus OOD làm một kết quả chung.

Workflow reference là teacher-derived: hard accuracy là agreement với teacher label, không phải kiểm chứng correctness độc lập bởi con người. Soft distributions được giữ và kiểm tra alignment. Yelp label 0..4 ứng với 1..5 sao, đánh giá rating được người viết gán từ review text. Public benchmarks có thể đã xuất hiện trong training.

## 3. Phương pháp và quy ước

### 3.1 Fit và split

Với shipped distribution $p$, fit thêm $t$ theo $p_t(k)=p(k)^{1/t}/\sum_jp(j)^{1/t}$. Tối thiểu hóa hard-reference NLL bằng bounded optimization trên log-temperature, $t\in[0.05,10]$. Đây là temperature bổ sung trên shipped probabilities, không mặc nhiên là total temperature của raw logits. Xác suất được clip ở 1e-12 khi tính log, và chuẩn hóa sai số rounding nhỏ.

20 seeds: 20261009–20261028. Mỗi workflow chọn 20% states cho fit; năm câu hỏi của một state luôn cùng tập. Mỗi split có 80 fit states/400 decisions và 320 eval states/1.600 decisions; giao theo state bằng 0. External evaluation không dùng external gold để fit T. Chỉ global và per-type được transfer OOD; per-task chỉ dùng khi có fitted workflow group, thiếu group phải raise. Leave-one-workflow-out fit ba workflow rồi eval workflow còn lại.

### 3.2 Metrics và uncertainty

NLL là fitting criterion; accuracy, mean confidence, multiclass Brier và top-label ECE được báo riêng. ECE chính có 15 equal-mass bins; sensitivity outputs có mass 5/10 và width 10/15. Equal-mass có thể tách tied confidences vào nhiều bins. Brier là sum over classes, binary range [0,2]. Temperature dương giữ argmax; lựa chọn đã lưu được bảo toàn ở tied predictions, nên accuracy không đổi.

Soft metrics: KL(reference||prediction), squared distribution distance và reference probability tại selected option. Ordinal RPS là mean squared cumulative-probability error trên K−1 internal thresholds; ordinal MAE là absolute error của expected rank, không phải MAE của selected argmax. Hard và soft targets có tên riêng trong CSV.

Bootstrap: 1.000 paired replicates, resample whole cases theo từng domain ở fit và eval, fit lại T trong mỗi replicate. Hai arms dùng cùng evaluation resample. Report 95% percentile intervals với Δ=after−before; âm tốt hơn cho loss/ECE. Raw-vs-shipped là fixed-T paired resampling; paired checkpoint comparisons dùng cùng external item resamples. 20 seed results là stability summaries trên cùng corpus, không phải 20 dataset độc lập. Intervals mang tính exploratory, không điều chỉnh multiple comparisons; không bao gồm training randomness hay uncertainty của lịch sử checkpoint.

## 4. Audit và các sửa đổi ảnh hưởng kết luận

| Vấn đề corpus/protocol cũ | Cách xử lý trong kết quả hiện tại |
|---|---|
| Decision-level split làm 266/400 cases có mặt ở cả fit và eval | Split theo state trước, tất cả questions cùng tập; test zero overlap |
| Per-task OOD dùng fallback T=1 nên transfer thực tế là no-op | Không xuất condition này; group thiếu raise |
| Gọi 5 checkpoints là 4 families | Dùng 3 release lineages: Liquid, OpenDecider, Jeff |
| “Score khó nhất ở 4/5 model” | Chỉ đúng với 2/5 theo ECE pooled; bỏ claim phổ quát |
| Gọi protocol cũ là exact replication paper gốc | Gọi empirical extension; khác corpus, fit/eval scheme và bins |
| Teacher labels được diễn giải như human correctness | Tách teacher agreement, soft fidelity và external author labels |
| Binary ties làm accuracy đổi sau scaling | Preserve stored argmax; kiểm tra accuracy invariance |
| Inference lỗi có thể bị bỏ qua nhưng run vẫn báo DONE | Resume theo unique key; expected/actual/errors/missing được manifest kiểm soát |
| Shared stored-temperature mechanism được gán cho cả hai d1 | Chỉ xác nhận explicit temperatures của pinned omni; không dựng raw arm cho 3B |
| Runtime/revision cũ thiếu metadata | Giữ giới hạn lịch sử; new runs có exact manifests/hashes |

### 4.1 Phần đã thực sự kiểm chứng

GPU inference do người dùng chạy trên Colab. Phần bàn giao này kiểm chứng returned data và source, tái tính metrics trên CPU và tạo phân tích bổ sung; không tuyên bố đã độc lập chạy lại model GPU. Cả hai new run đủ 5.000/5.000 keys; không có missing hoặc inference error được ghi nhận. Prediction trong checkpoint riêng và full export khớp từng byte. Labels, hard/soft gold, finite/normalized distributions và exact case SHA256 đều được kiểm tra.

Exact normalized-text duplicate check không thấy trùng trong 3.400 cases; điều này chỉ loại trùng theo lowercase/whitespace, không loại near duplicates hoặc training exposure. Tất cả 20 splits/model đã kiểm tra no fit/eval case overlap. Source inference/analysis được export khớp code bàn giao trước đó; as-shipped metric recomputation khớp numeric outputs tới sai số float. Hai binary ties của 3B giữ stored choice. Typed selections ở cả hai d1 không đổi so với corpus cũ, max probability differences khoảng 1.4e-6.

Tests gồm regression trên 11.000 historical distributions, case split/tie/metric equivalence/soft gold; fake-adapter integration test kiểm tra coverage manifest, inverse-temperature export, resume và reject changed config. Fake adapter chỉ kiểm tra logic IO, không phải bằng chứng về GPU/model quality. Gói đã được giải nén ở thư mục tạm và chạy tests, report generation và plotting thành công.

## 5. Kết quả Colab mới theo task

Phần sau giữ đầy đủ bảng và intervals của completed-run report. Các hướng “cách cập nhật paper” đã được thực hiện trong PAPER_DRAFT.md; các điều kiện cross-family nêu ở đây vẫn là giới hạn hiện tại.

## Kết quả Colab đã xác minh — hai checkpoint d1

**Cả hai run đạt kiểm tra dữ liệu và tái tính metrics.** d1-omni và d1-3B đều đủ 5.000/5.000 quyết định, không thiếu mẫu/không có inference error được ghi nhận. Checkpoint riêng và ZIP tổng có prediction giống nhau từng byte. Không cần chạy lại hai model này để sửa hoặc bổ sung dữ liệu của lượt mặc định.

### Những kiểm tra đã hoàn thành

- Hash của exact `cases.jsonl` khớp hai manifest: `318f199b56cdc08d6db9865edbf73d8264017fe48dc909a550f7fde6aa54ee43`.
- Hai model cùng 3.400 case, gồm 400 workflow case × 5 câu hỏi và 3.000 OOD case × 1 câu hỏi.
- Đủ khóa `(domain, case_id, question)`, không trùng prediction; valid option/gold/probabilities; soft gold trong CSV khớp input snapshot. Không có text trùng sau chuẩn hóa lowercase/whitespace trong 3.400 case; đây không phải kiểm tra nhiễm dữ liệu training.
- Tái tính toàn bộ as-shipped metrics từ CSV: khớp bảng Colab tới sai số số thực. Code phân tích và inference được export khớp source đã bàn giao.
- Kiểm tra 20 seed/model: fit/eval không có case chung. Temperature giữ accuracy, kể cả hai trường hợp binary tie của d1-3B.
- d1-omni raw arm khớp phép đảo temperature explicit của cùng snapshot ở mọi dòng. Không dựng raw arm cho d1-3B.
- Tính thêm metrics theo từng task; 1.000 paired bootstrap/task (refit ID temperatures trong từng replicate); 20 calibration splits; paired model comparisons trên cùng test items.

Metadata Colab ghi GPU T4, FP16 theo source đã chạy. Vòng inference sau model loading mất khoảng 108 giây (omni) và 306 giây (3B); số này không bao gồm download/load và không phải benchmark latency kiểm soát.

### As-shipped trên OOD cân bằng

| Model | Task | n | Accuracy | Mean confidence | ECE15 | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| d1-omni | AG News | 1000 | 0.8710 | 0.8743 | 0.0213 | 0.3812 | 0.1945 |
| d1-omni | Rotten Tomatoes | 500 | 0.8760 | 0.9397 | 0.0651 | 0.3412 | 0.1999 |
| d1-omni | SST-2 | 500 | 0.9260 | 0.9468 | 0.0350 | 0.2363 | 0.1274 |
| d1-omni | Yelp rating | 1000 | 0.6340 | 0.6350 | 0.0418 | 0.8108 | 0.4675 |
| d1-3b | AG News | 1000 | 0.9030 | 0.9106 | 0.0255 | 0.3199 | 0.1578 |
| d1-3b | Rotten Tomatoes | 500 | 0.8880 | 0.8890 | 0.0339 | 0.2736 | 0.1645 |
| d1-3b | SST-2 | 500 | 0.9520 | 0.9111 | 0.0446 | 0.1712 | 0.0889 |
| d1-3b | Yelp rating | 1000 | 0.6310 | 0.6850 | 0.0632 | 0.8264 | 0.4896 |

Yelp cân bằng 200 review/mỗi mức, 5 mức. Level 0..4 tương ứng 1..5 sao; không cộng hoặc trừ nhãn ngoài mapping này. Hai model có accuracy Yelp gần nhau; không suy ra 3B tốt hơn trên rating chỉ từ số tham số. Paired model differences và CI được cung cấp trong bundle.

### Kết quả quan trọng nhất: refit trong workflow không đảm bảo transfer sang rating

Bảng sau là trung bình qua 20 calibration splits theo case. “Per-type” dùng riêng T fit trên workflow score để áp cho Yelp score; accuracy giữ nguyên.

| Model | Mode | Yelp ECE shipped→refit | Yelp NLL shipped→refit | Số split NLL xấu hơn |
|---|---|---:|---:|---:|
| d1-omni | global | 0.0418→0.0682 | 0.8108→0.8365 | 20/20 |
| d1-omni | per_type | 0.0418→0.1239 | 0.8108→0.8881 | 20/20 |
| d1-3b | global | 0.0632→0.0483 | 0.8264→0.8254 | 4/20 |
| d1-3b | per_type | 0.0632→0.0550 | 0.8264→0.8269 | 10/20 |

Với **d1-omni**, workflow refit giúp ID nhưng làm model thiếu tự tin trên Yelp vốn có mean confidence gần đúng accuracy. Với **d1-3B**, ECE và NLL có thể đi theo hai hướng khác nhau; không gọi giảm ECE là cải thiện mọi khía cạnh của phân phối xác suất.

Các interval dưới đây dùng seed chính 20261009, nên point delta khác trung bình 20 seed. Delta âm là giảm loss/ECE. Bootstrap có refit temperature, không chỉ lấy lại test samples.

| Model | Yelp comparison | Metric | Δ | 95% percentile interval |
|---|---|---|---:|---|
| d1-omni | per-type refit − shipped | ece_mass_15 | 0.0737 | [0.0239, 0.1218] |
| d1-omni | per-type refit − shipped | nll | 0.0678 | [0.0278, 0.1238] |
| d1-3b | per-type refit − shipped | ece_mass_15 | -0.0363 | [-0.0456, 0.0165] |
| d1-3b | per-type refit − shipped | nll | 0.0035 | [-0.0069, 0.0322] |

### Raw/shipped ablation của d1-omni

Explicit shipped temperature đã giảm lỗi trên cả bốn OOD task trong lượt này. Riêng Yelp:

| Arm | ECE15 | NLL | Brier |
|---|---:|---:|---:|
| raw_reconstructed | 0.1203 | 0.8759 | 0.4910 |
| shipped | 0.0418 | 0.8108 | 0.4675 |
| ID_refit_per_type | 0.1155 | 0.8786 | 0.4930 |

Đây là ablation trực tiếp phép temperature trên cùng predictions/weights và cùng inputs, không chứng minh nguyên nhân training/architecture. Raw được khôi phục bằng phép đảo temperature đã kiểm tra với source, có giới hạn precision. T additional fit trên workflow nhân với T shipped; không được mô tả T additional như nhiệt độ tuyệt đối của logits gốc.

### Cách cập nhật paper

Luận điểm phù hợp: **calibration theo primitive vẫn phụ thuộc task; temperature sửa workflow có thể làm hỏng ordinal OOD vốn được calibrate tốt.** Đây là kết quả thực nghiệm cụ thể cho d1-omni và bộ Yelp được chọn, không phải định lý mới hoặc quy luật mọi decision model.

- Giữ bảng 5 checkpoint/3 release families từ corpus cũ làm exploratory cross-family context.
- Dùng hai d1 × 3.000 OOD mới làm extension có checkpoint hashes, exact input snapshot, class balance và ordinal metrics.
- Không gộp OOD 200 mẫu cũ với OOD 3.000 mới để khẳng định một xu hướng chung: câu chuyện transfer của omni đổi khi tập test lớn hơn và có score.
- Main finding mạnh hơn trên Yelp là cả ECE và NLL của omni đều xấu đi sau ID per-type refit, với percentile intervals dương ở split chính và 20/20 split xấu hơn.
- Workflow score gồm 700 quyết định 4 mức và 100 quyết định 5 mức; Yelp đều 5 mức. Shift thay cả semantics/rubric và phân bố số lựa chọn, nên chưa cô lập nguyên nhân do domain hay cardinality.
- Bootstrap là exploratory, không điều chỉnh multiple comparisons. Không dùng 20 seed như 20 dataset độc lập.

Có thể hoàn thiện bản thảo với các kết quả đã nhận. Nếu mở rộng claim semantic OOD **cross-family**, cần chạy ba checkpoint OpenDecider/Jeff trên đúng `cases_sha256`; hiện chúng chỉ có OOD nhỏ cũ. Không cần thêm inference chỉ để viết xong phần hai d1. Author/affiliation và venue vẫn cần chốt trước submission; teacher gold và novelty relative to prior work phải được trình bày đúng.

### Files và tái hiện

Bundle cập nhật giữ dữ liệu, source đã chạy, analysis theo task, bootstrap, paired model CI và bản thảo. `taskwise_metrics.csv` có ordinal RPS/MAE hard-target; dữ liệu typed có thêm soft-target metrics.

Chạy lại CPU analysis không dùng GPU. Dùng `verified/scripts/verify_analysis.py --models d1-omni d1-3b --data-dir colab_complete/extension_output`. Analysis bổ sung theo task có scripts riêng trong `checkpoint_omni/` và `colab_complete/`.


## 6. Kết quả corpus lịch sử — phân tích lại

Toàn bộ bảng dưới đây chỉ dùng corpus lịch sử: OOD n=200/model. Không diễn giải chúng như kết quả trên 3.000 OOD items mới.

## Verified calibration results

Delta = after − before. Negative losses/ECE indicate improvement. Repeated splits are dependent stability measurements; their spread is not a confidence interval. Bootstrap intervals resample cases and refit temperatures.

### As shipped: whole datasets

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

### Repeated case-disjoint calibration splits (all types pooled)

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

### Paired cluster bootstrap (seed 0; fit refitted each replicate)

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

### Leave-one-workflow-out (per-type, score questions only)

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


## 7. Truy vết kết luận → evidence → code

Đường dẫn dưới đây tương đối với thư mục giải nén ZIP.

| Nội dung cần kiểm chứng | Evidence/data | Code |
|---|---|---|
| Inputs, số rows và provenance của hai d1 | `colab_complete/extension_output/cases.jsonl`, `manifest_d1-*.json`, `config_d1-*.json`, `resolved_revisions.json`; `run_status.json`, `notebook_config.json`, `pip_freeze.txt`, `logs/` | `verified/scripts/run_extension.py`; source exact run trong `colab_complete/executed_source/` |
| As-shipped metrics và case-disjoint refit mới | `colab_complete/extension_output/predictions_d1-*.csv`; `colab_complete/extension_results/` | `verified/scripts/verify_analysis.py`, `summarize_verified.py` |
| Omni taskwise harm và raw/shipped arm | `checkpoint_omni/taskwise/{audit.json,taskwise_metrics.csv,taskwise_repeated_splits.csv,taskwise_bootstrap.csv,fit_temperatures_seed0.json}`; raw CSV trong `extension_output/` | `checkpoint_omni/analyze_checkpoint.py` |
| 3B taskwise result | `colab_complete/taskwise_d1-3b/` gồm cùng loại files | `colab_complete/analyze_tasks.py` |
| Paired checkpoint differences | `colab_complete/paired_model_differences.csv` | `colab_complete/paired_models.py` |
| Historical five-checkpoint context | `verified/data/`, `verified/provenance/`, `verified/results_corrected/` | `verified/scripts/verify_analysis.py`, `summarize_verified.py`, `test_verified.py` |
| Figures mới | `colab_complete/taskwise_temperature_effects.png` và `.pdf` | `colab_complete/make_task_figures.py` |
| Report mới và report tổng hợp | `COLAB_RESULTS_VERIFIED.md`, `REPORT_FINAL.md` | `colab_complete/write_results_report.py`, `write_final_report.py` |
| Integrity của gói | `PACKAGE_MANIFEST.json`: file path, SHA256, byte size | `colab_complete/package_results.py`; lệnh kiểm tra bên dưới |

Bundle có đầy đủ prediction/input snapshot/source phân tích để tái hiện các kết quả. Model weights và toàn bộ third-party remote repository không được nhúng vào ZIP; chúng được resolve từ model IDs/revisions khi chạy inference. New source exact-run snapshot được giữ nguyên, có thể còn chứa README/bản thảo lúc trước khi nhận output; trạng thái hiện tại lấy từ root REPORT_FINAL.md/PAPER_DRAFT.md. Các LEGACY_* và VERIFY_REPORT.md là lịch sử audit, không phải trạng thái Colab mới. Không dùng các pending instructions trong audit cũ như yêu cầu inference còn thiếu hôm nay.

## 8. Tái hiện kết quả

### 8.1 CPU: không cần GPU hoặc tải weights

Chạy từ thư mục giải nén ZIP. Để giữ outputs bàn giao, dùng một bản giải nén riêng khi chạy scripts taskwise vì chúng ghi vào thư mục cố định.

```bash
python -m pip install numpy scipy matplotlib
python verified/scripts/test_verified.py
python verified/scripts/test_extension_io.py
python verified/scripts/verify_analysis.py --data-dir colab_complete/extension_output --models d1-omni d1-3b --seeds 20 --bootstrap 1000 --out reproduced_extension
python verified/scripts/summarize_verified.py reproduced_extension
python verified/scripts/make_verified_figures.py reproduced_extension
python checkpoint_omni/analyze_checkpoint.py --bootstrap 1000
python colab_complete/analyze_tasks.py --bootstrap 1000
python colab_complete/paired_models.py
python colab_complete/make_task_figures.py
python colab_complete/write_results_report.py
python colab_complete/write_final_report.py
```

Historical reanalysis:

```bash
python verified/scripts/verify_analysis.py --seeds 20 --bootstrap 1000 --out reproduced_historical
python verified/scripts/summarize_verified.py reproduced_historical
```

`pip_freeze.txt` và run manifests ghi môi trường inference thực tế. Dependency ranges trong requirements không đảm bảo mọi phiên bản tương lai cho cùng kết quả; scientific Python phiên bản khác có thể tạo sai số số thực nhỏ. Manifest/source/cases được giữ để kiểm tra protocol và checkpoint, không chỉ final scalar scores.

### 8.2 Kiểm tra archive trước khi giải nén

Đặt ZIP tại current directory rồi chạy:

```python
import hashlib, json, zipfile

with zipfile.ZipFile('decision-calibration-verified-v2.zip') as z:
    assert z.testzip() is None
    m = json.loads(z.read('PACKAGE_MANIFEST.json'))
    assert set(z.namelist()) == set(m['files']) | {'PACKAGE_MANIFEST.json'}
    for name, meta in m['files'].items():
        data = z.read(name)
        assert len(data) == meta['bytes'], name
        assert hashlib.sha256(data).hexdigest() == meta['sha256'], name
    print('PASS CRC + SHA256:', len(m['files']), 'files')
```

Checksum kiểm tra consistency/corruption của gói, không tự chứng minh chất lượng labels hoặc kết luận khoa học. Manifest không tự hash chính nó.

### 8.3 Colab inference

`Decision_Calibration_Verified_Colab.ipynb` là notebook self-contained đã dùng cho các runs hoàn tất. Mặc định hai d1, FP16, subprocess/model; inputs/source/data được nhúng. Không cần chạy lại mặc định để hoàn thành paper. Nếu tái chạy hoặc mở rộng, cần giữ exact case snapshot và immutable resolved revisions; probe là smoke check, không trộn probe vào bảng full run. Historical predictions không thể được hồi tố gán provenance của snapshot mới.

## 9. Giới hạn và việc còn lại trước submission

1. Main semantic OOD extension chỉ có hai d1; ba checkpoint còn lại chỉ có small probes lịch sử. Không suy rộng sang mọi release family.
2. Workflow score gồm 700 decisions K=4 và 100 K=5; Yelp đều K=5. Task/rubric và option-count mix cùng thay đổi; chưa cô lập causal driver.
3. Workflow gold tổng hợp, rating chủ quan; không có independent human adjudication mới. Public benchmark exposure chưa được loại trừ.
4. Raw probabilities của omni được inverse-transform từ shipped outputs của cùng snapshot, có giới hạn clipping/float precision. Đây là probability-operation ablation, không tách training/architecture causes; không có reconstructed raw arm cho 3B.
5. Bootstrap exploratory, ECE phụ thuộc bins/ties/sample size; không dùng ECE một mình để tuyên bố calibrated hoàn hảo. Pooled scores có thể che task-specific failure; report giữ cả taskwise metrics.
6. Novelty nằm ở empirical extension cụ thể và evidence, không ở phương pháp temperature scaling. Cần chốt authors/affiliations, venue format và rà soát related work trước submission; đây là việc hoàn thiện manuscript, không phải mặc nhiên yêu cầu thêm GPU runs.

## 10. Các tài liệu đọc cùng

- `PAPER_DRAFT.md`: manuscript tiếng Anh cập nhật đầy đủ kết quả hiện tại.
- `COLAB_RESULTS_VERIFIED.md`: report ngắn về hai completed d1 runs.
- `VERIFY_REPORT.md`: audit ban đầu và các lỗi đã sửa, giữ nguyên để truy vết lịch sử.
- `verified/LEGACY_*`: reports/source claims gốc; chỉ để đối chiếu.
- `README.md`: cấu trúc và lệnh tái hiện của gói.

Các nguồn học thuật và model/dataset links đã đối chiếu được ghi trong References của PAPER_DRAFT.md và audit ban đầu. Báo cáo này tổng hợp evidence từ outputs được nhận và CPU analysis, không bổ sung tuyên bố rằng đã thực hiện một literature search mới ở lượt bàn giao này.
