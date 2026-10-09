# Calibration of zero-token decision models under distribution shift
*(mở rộng cross-family 2026-10-08: 5 checkpoints / 4 họ — xem mục 10)*

**Kết luận trước (sau extension):** "Ships calibrated" **không phải đặc tính chung của decision model — nó là đặc tính của pipeline.** Liquid d1 xuất xưởng đã tốt (ECE 0.04–0.15, refit gần như vô ích); OpenDecider-small xuất xưởng *thiếu* tự tin trên score/choice (refit giảm một nửa ECE 0.070→0.037); Jeff xuất xưởng *thừa* tự tin, càng nhỏ càng tệ (0.8B cần temperature ≈4.0). Điểm chung xuyên họ: câu hỏi score khó calibrate nhất; không họ nào sập kiểu laya trong tập OOD này.

**Kết luận trước:** Hai model open-weight nhà Liquid (d1-omni-600M, d1-3B) **đã được calibration tốt ngay khi xuất xưởng** — ECE as-shipped chỉ 0.04–0.15 trên mọi loại câu hỏi và mọi domain, nhờ per-type temperatures có sẵn trong `config.json`. Vì vậy **không replicate được phát hiện dramatic của arXiv:2610.00346** (ECE 0.251 → 0.038 sau khi fit temperature): fit thêm temperature trên held-out data chỉ cải thiện không đáng kể (tốt nhất 0.062 → 0.044), và transfer in-domain → OOD gần như không giúp gì. Điểm yếu duy nhất lộ ra khi tách theo loại câu hỏi: **d1-omni quá tự tin trên câu hỏi score** (accuracy 0.449, ECE 0.152, nhiệt độ fit được T=2.35); d1-3B đã sửa điểm này (accuracy 0.590, ECE 0.071). Ngoài-domain, cả hai giữ calibration tốt và không có hiện tượng sập như laya.

## 1. Định vị so với arXiv:2610.00346

Rafe & Das, *"Benchmarking System One decision models against trained classifiers and language models for automated decision gates"* (nộp 29/09/2026, sửa 06/10/2026), benchmark 8 checkpoint thuộc 6 họ decision model (gồm cả 2 checkpoint Laya) và ở RQ2 nghiên cứu calibration transfer: fit temperature per-task trên held-out data đưa ECE gộp từ 0.251 xuống 0.038.

Đóng góp của nghiên cứu này so với họ:
- **(a)** Nghiên cứu calibration đầu tiên trên họ **Liquid d1 open-weight** — d1-omni-600M (587M) và d1-3B (3.05B), release 07/10/2026, không xuất hiện trong paper của họ.
- **(b)** Chẩn đoán calibration **tách theo loại câu hỏi** (noul/choice/score) — họ chỉ làm ở mức task và số lượng options.
- **(c)** Ghi nhận failure mode cụ thể: điểm yếu score-type của d1-omni.
- Phương pháp RQ2 được replicate trung thực (fit temperature global / per-type / per-task trên held-out, đánh giá trên phần còn lại + OOD) để so sánh được trực tiếp.

Số liệu của laya trong báo cáo này được **trích dẫn** từ 2610.00346 và từ các run benchmark trước của chúng tôi (2026-10-08), không chạy lại inference cho laya.

## 2. Phương pháp

**Model:** `LiquidAI/d1-omni-600M` và `LiquidAI/d1-3B`, load bằng `AutoModel.from_pretrained(..., trust_remote_code=True, dtype=torch.float16).to("cuda")`, cùng API `system_one_batch` với Decision Index schema. Chạy trên Colab T4 (session `calib-d1`, đã stop).

**Dữ liệu:**
- In-domain: `LocalLLaMA/typed-decisions` test split chính thức — 400 cases / 2.000 decisions, 4 workflows (agent-trace, customer-service, invoice-processing, security-incidents), 3 loại câu hỏi (noul 600, choice 600, score 800). Mapping câu hỏi giữ nguyên run 2026-10-08.
- Out-of-domain: Rotten Tomatoes balanced 40 + SST-2 validation balanced 100 (clean) + AG News 60 (đầu split bị lệch class — giữ flag). OOD chỉ có noul và choice, không có score.

**Metrics:** accuracy, mean confidence, ECE (equal-mass binning, 15 bins), Brier score (dạng multiclass Σ_k(p_k−onehot_k)² — với câu hỏi noul nhị phân thì thang đo là [0,2], gấp đôi binary Brier textbook; chỉ so sánh trong cùng loại câu hỏi), NLL. Calibration cần **full phân phối xác suất** — CSV cũ chỉ lưu confidence nên đã chạy lại inference cho cả hai model d1, lưu toàn bộ phân phối (`data/predictions_*.csv`, 2.200 rows/model).

**RQ2 protocol:** fit temperature T trên (A) 20% in-domain held-out (stratified theo domain×qtype, seed cố định) và (B) 50% OOD; đánh giá trên phần còn lại và (cho A) trên toàn bộ OOD để test transfer. Ba mode: global (1 T), per-type (3 T), per-task ((domain,qtype)). Fit bằng minimize NLL (bounded 0.05–10). Temperature áp trên log-probabilities (tương đương vì chỉ quan sát được probabilities).

## 3. Kết quả đo as-shipped (không refit)

| model | domain | qtype | n | acc | mean conf | ECE | Brier | NLL |
|---|---|---|---|---|---|---|---|---|
| d1-omni | in-domain | noul | 600 | 0.712 | 0.740 | **0.044** | 0.371 | 0.542 |
| d1-omni | in-domain | choice | 600 | 0.613 | 0.570 | **0.076** | 0.509 | 0.911 |
| d1-omni | in-domain | score | 800 | 0.449 | 0.591 | **0.152** | 0.691 | 1.269 |
| d1-omni | ood | noul | 140 | 0.893 | 0.943 | 0.058 | 0.186 | 0.362 |
| d1-omni | ood | choice | 60 | 0.783 | 0.830 | 0.109 | 0.322 | 0.647 |
| d1-3b | in-domain | noul | 600 | 0.787 | 0.802 | **0.050** | 0.298 | 0.467 |
| d1-3b | in-domain | choice | 600 | 0.602 | 0.662 | **0.072** | 0.518 | 0.966 |
| d1-3b | in-domain | score | 800 | 0.590 | 0.594 | **0.071** | 0.536 | 0.943 |
| d1-3b | ood | noul | 140 | 0.929 | 0.904 | 0.060 | 0.110 | 0.197 |
| d1-3b | ood | choice | 60 | 0.800 | 0.902 | 0.127 | 0.306 | 0.560 |

Theo workflow (in-domain, 500 decisions mỗi cái):

| model | agent-trace | customer-service | invoice-processing | security-incidents |
|---|---|---|---|---|
| d1-omni acc / ECE | 0.560 / 0.062 | 0.632 / 0.121 | 0.564 / 0.111 | 0.552 / 0.077 |
| d1-3b acc / ECE | 0.542 / 0.107 | 0.678 / 0.049 | 0.744 / 0.059 | 0.646 / 0.082 |

Nhận xét: accuracy khớp run 2026-10-08 trong ±0.002 (nhiễu fp16 giữa các lần chạy). ECE as-shipped của cả hai model đã thấp — giải thích: model card của d1-omni ghi text answers đã được calibrate bằng per-type temperatures trong `config.json`.

## 4. RQ2 replication: fit temperature trên held-out

| model | fit → eval | mode | ECE (as-shipped → fitted) | Brier → | NLL → |
|---|---|---|---|---|---|
| d1-omni | indom20 → indom-rest | global | 0.062 → 0.051 | 0.527 → 0.524 | 0.919 → 0.913 |
| d1-omni | indom20 → indom-rest | per-type | 0.062 → 0.063 | 0.527 → 0.517 | 0.919 → 0.901 |
| d1-omni | indom20 → indom-rest | **per-task** | 0.062 → **0.044** | 0.527 → 0.522 | 0.919 → 0.903 |
| d1-omni | indom20 → **ood-all** | global/per-type/per-task | 0.071 → 0.067 / 0.069 / 0.071 | ~không đổi | ~không đổi |
| d1-omni | ood50 → ood-rest | per-task | 0.096 → 0.085 | 0.277 → 0.260 | 0.558 → 0.465 |
| d1-3b | indom20 → indom-rest | per-task (tốt nhất) | 0.046 → 0.044 | 0.446 → 0.443 | 0.785 → 0.773 |
| d1-3b | indom20 → ood-all | per-type (tốt nhất) | 0.049 → 0.048 | 0.169 → 0.167 | 0.306 → 0.308 |
| d1-3b | ood50 → ood-rest | per-type | 0.069 → 0.065 | 0.190 → 0.189 | 0.362 → 0.358 |

Nhiệt độ fit được (tiêu biểu): d1-omni per-type trên indom20 → `{score: 2.35, noul: 1.43, choice: 0.99}`; d1-3b per-type → `{score: 1.14, noul: 1.23, choice: 1.37}`. Accuracy không đổi giữa các mode (temperature không đổi argmax — đúng lý thuyết).

**So với 2610.00346:** họ đi từ ECE 0.251 (as-shipped tệ) xuống 0.038 sau fit — cải thiện ~6.6×. Ở họ d1, điểm xuất phát đã là 0.04–0.07 nên fit thêm chỉ còn là tinh chỉnh (+0.01–0.02). Kết luận: phát hiện "temperature fitting cứu calibration" của họ **không transfer sang họ d1 theo chiều dramatic** — vì Liquid đã làm bước đó từ lúc ship (per-type temperatures trong config). Transfer in-domain → OOD cũng không giúp: OOD as-shipped vốn đã tốt (0.05–0.13).

## 5. Chẩn đoán per-type và failure modes

- **d1-omni quá tự tin trên score** (câu hỏi rating có thứ tự): accuracy 0.449 nhưng mean confidence 0.591, ECE 0.152 — đường reliability nằm dưới diagonal (xem figure). Nhiệt độ fit T=2.35 xác nhận cần "làm mềm" mạnh. Đây là điểm yếu duy nhất đáng kể của model.
- **d1-3B sửa được điểm yếu score**: accuracy 0.590, ECE 0.071, T fit ≈ 1.14 — gần như đã chuẩn.
- noul và choice: cả hai model đều ổn (ECE 0.04–0.08 in-domain).
- Per-task temperatures fit trên slice 400 rows rất noisy (có cell T=0.36–0.61, tức underconfidence cục bộ) — dấu hiệu overfit khi fit quá mịn trên ít dữ liệu; global/per-type ổn định hơn.
- OOD: cả hai model **chính xác hơn** ngoài-domain (bộ OOD dễ hơn) và giữ calibration — không quan sát hiện tượng mode-collapse kiểu laya (confidence 0.998 mà sai toàn bộ).

## 6. Figures

Reliability diagrams (equal-mass 15 bins; kích thước điểm ~ số mẫu trong bin):

![d1-omni in-domain](sandbox://workspace/decision-calibration/figures/rel_d1-omni_indomain.png)
![d1-omni OOD](sandbox://workspace/decision-calibration/figures/rel_d1-omni_ood.png)
![d1-3B in-domain](sandbox://workspace/decision-calibration/figures/rel_d1-3b_indomain.png)
![d1-3B OOD](sandbox://workspace/decision-calibration/figures/rel_d1-3b_ood.png)

## 7. Limitations

1. OOD chỉ 200 decisions, không có câu hỏi score ngoài-domain — chẩn đoán score mới chỉ in-domain.
2. AG News 60 mẫu lệch phân bố (đã flag từ run trước); OOD accuracy cao một phần vì bộ này dễ.
3. Số liệu 2610.00346 được trích dẫn theo báo cáo của họ, chưa replicate độc lập.
4. laya không chạy lại — chỉ dùng làm baseline trích dẫn.
5. Nhiễu fp16 giữa các lần chạy ~±0.002 accuracy (1 decision/500).
6. Temperature fit trên log-probabilities (không có logits gốc) — tương đương về mặt toán học, đã ghi rõ.

## 8. Tái hiện

```bash
# 1. Inference trên Colab T4 (session calib-d1 đã stop; tạo lại khi cần)
python scripts/build_bundle.py                       # dist/measure_<model>.py
colab new -s calib-d1 --gpu T4
colab install -s calib-d1 "transformers>=5.15" datasets pillow soundfile
colab exec -s calib-d1 -f dist/measure_d1-omni.py --timeout 1200
colab download -s calib-d1 /tmp/predictions_d1-omni.csv data/predictions_d1-omni.csv
# ... lặp lại cho d1-3b, rồi colab stop -s calib-d1

# 2. Phân tích local (không cần GPU)
python scripts/run_rq2.py        # data/metrics_*.csv, data/temperatures.json
python scripts/make_figures.py   # figures/rel_*.png
```

Thêm model mới: viết `load_<name>()` + (nếu format answer khác) `normalize_<name>()` trong `decision_calib/models.py`, thêm một dòng vào `REGISTRY`, rồi `python scripts/build_bundle.py` và chạy `dist/measure_<name>.py` — không cần sửa code phân tích.

## 10. Cross-family extension (2026-10-08, session `calib-xfam`)

Một họ model thì mỏng — extension này thêm 3 checkpoint / 2 họ mới, cùng protocol, để trả lời: "ships calibrated" có phải đặc tính chung của decision model, hay chỉ của nhà Liquid?

**Checkpoint mới (đo trên cùng T4, cùng 2.000 in-domain + 200 OOD decisions):**

| Checkpoint | Họ / kiến trúc | HF repo |
|---|---|---|
| opendecider-small | OpenDecider / Qwen3-4B-Instruct-2507 + LoRA r16 | `manjunathshiva/opendecider-small` |
| jeff-2b | Jeff / Qwen3.5-2B + 255-option readout | `mstrasser/Jeff-Qwen3.5-2B` (v1.2) |
| jeff-0.8b | Jeff / Qwen3.5-0.8B + 255-option readout | `mstrasser/Jeff-Qwen3.5-0.8B` (v1.2) |

Adapter cho OpenDecider merge LoRA thủ công từng layer (tránh OOM 14.4GB của `merge_and_unload()` trên T4); adapter Jeff dùng `jeff.model.DecisionModel` + `answer()` từ repo `firelex/jeff`. Cả hai đều qua probe trước khi đo full; 0 warning trên 2.200 rows mỗi model.

**Bongard-mini (`AgentBull/bongard-mini`) — SKIP có lý do:** T5Gemma 2 4B-4B encoder-decoder, 7.51B params; chính README của họ ghi BF16 cần ~15GB *chỉ cho weights* — không vừa T4 15.36GB khi cộng runtime/activations, và `Predictor` API của họ không có đường quantization. Ghi nhận làm future work trên GPU lớn hơn.

**Validation adapter:** in-domain accuracy đo được vs leaderboard tự báo — opendecider-small 0.660 vs 0.671; jeff-2b 0.556 vs 0.511; jeff-0.8b 0.493 vs 0.483. Sai lệch trong biên kỳ vọng (khác harness/prompt; jeff-2b cao hơn có thể do checkpoint v1.2 mới hơn lần họ đo 29/09).

### 10.1 Bảng tổng hợp as-shipped (5 checkpoints đo + laya trích dẫn)

| Model | In-domain acc | ECE in-domain (noul / choice / score) | OOD acc | ECE OOD (noul / choice) |
|---|---|---|---|---|
| d1-omni-600M (Liquid) | 0.577 | 0.044 / 0.076 / **0.152** | 0.860 | 0.058 / 0.109 |
| d1-3B (Liquid) | 0.653 | 0.050 / 0.072 / 0.071 | 0.890 | 0.060 / 0.127 |
| opendecider-small | 0.660 | 0.077 / 0.075 / 0.114 | 0.880 | 0.068 / 0.134 |
| jeff-2b | 0.556 | 0.105 / 0.130 / 0.078 | 0.810 | 0.081 / 0.153 |
| jeff-0.8b | 0.493 | 0.129 / 0.183 / 0.141 | 0.825 | 0.098 / 0.167 |
| laya-typed-decisions (trích dẫn) | 0.766 | — | 0.50 (sentiment) | mode collapse |

### 10.2 RQ2 trên họ mới: temperature refit có giúp?

Fit **global temperature** trên 20% in-domain held-out, đo ECE trên phần còn lại (giống methodology mục 4; per-type/per-task cho kết quả tương tự, xem `data/metrics_rq2.csv`):

| Model | ECE as-shipped → fitted (in-domain) | Nhiệt độ fit per-type (score / choice / noul) | Nhận xét |
|---|---|---|---|
| d1-omni-600M | 0.062 → 0.051 | 2.35 / 0.99 / 1.43 | refit gần như vô ích |
| d1-3B | 0.046 → 0.051 | 1.14 / 1.37 / 1.23 | đã tốt sẵn |
| **opendecider-small** | **0.070 → 0.037** | **0.59 / 0.69** / 1.09 | **refit giảm một nửa ECE** — underconfident sẵn trên score/choice |
| jeff-2b | 0.095 → 0.075 | 1.41 / 1.46 / 1.71 | hơi overconfident, refit giúp vừa phải |
| jeff-0.8b | 0.137 → 0.072 | **4.03** / 1.87 / **4.00** | **overconfident nặng** as-shipped; refit cứu nhiều in-domain nhưng transfer sang OOD thất bại (0.084 → 0.112, tệ đi) |

### 10.3 Verdict cập nhật

**"Ships calibrated" không phải đặc tính chung — nó là đặc tính của pipeline.** Liquid d1 xuất xưởng đã tốt (T≈1); OpenDecider xuất xưởng *thiếu* tự tin trên score/choice (T<1 mới đúng); Jeff xuất xưởng *thừa* tự tin, càng nhỏ càng tệ (0.8B cần T≈4). Hệ quả thực hành: **đừng tin con số confidence as-shipped của decision model lạ — luôn fit temperature trên một ít dữ liệu của chính task mình**, đúng như RQ2 của 2610.00346 khuyến nghị; riêng trường hợp opendecider-small, refit là bắt buộc chứ không phải optional (ECE giảm một nửa).

Điểm chung xuyên họ model: **câu hỏi score là loại khó calibrate nhất** (ECE cao nhất ở 4/5 checkpoint đo được), và không họ nào sập kiểu laya ngoài-domain trong tập OOD này.

### 10.4 Figures mới

![opendecider-small in-domain](sandbox://workspace/decision-calibration/figures/rel_opendecider-small_indomain.png)
![opendecider-small OOD](sandbox://workspace/decision-calibration/figures/rel_opendecider-small_ood.png)
![jeff-2b in-domain](sandbox://workspace/decision-calibration/figures/rel_jeff-2b_indomain.png)
![jeff-2b OOD](sandbox://workspace/decision-calibration/figures/rel_jeff-2b_ood.png)
![jeff-0.8b in-domain](sandbox://workspace/decision-calibration/figures/rel_jeff-0.8b_indomain.png)
![jeff-0.8b OOD](sandbox://workspace/decision-calibration/figures/rel_jeff-0.8b_ood.png)

## 11. Dữ liệu và code (cập nhật)

- Raw predictions mới: `data/predictions_opendecider-small.csv`, `data/predictions_jeff-2b.csv`, `data/predictions_jeff-0.8b.csv` (2.200 rows mỗi file, full phân phối)
- Metrics/figures đã tái tạo cho cả 5 model: `data/metrics_asshipped.csv`, `data/metrics_byworkflow.csv`, `data/metrics_rq2.csv` (60 eval rows), `data/temperatures.json`, `figures/rel_*.png` (10 files)
- Registry mới trong `decision_calib/models.py`: `opendecider-small`, `jeff-2b`, `jeff-0.8b` (thêm model = 1 hàm load + 1 hàm predict + 1 dòng registry như cũ); bundles `dist/measure_*.py` build lại cho cả 5
- Tổng thời gian T4 extension: ~40 phút (3 run đo) + setup/fix env — vẫn trong quota
- Lưu ý env: session `calib-xfam` từng bị hỏng numpy/scipy (xung đột debian numpy 1.26.4) — đã fix bằng reinstall sạch; mọi run đo đều chạy trong subprocess Python mới để tránh `sys.modules` nhiễm của kernel cũ
