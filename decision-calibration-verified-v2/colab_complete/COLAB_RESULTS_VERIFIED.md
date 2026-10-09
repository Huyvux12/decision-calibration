# Kết quả Colab đã xác minh — hai checkpoint d1

**Cả hai run đạt kiểm tra dữ liệu và tái tính metrics.** d1-omni và d1-3B đều đủ 5.000/5.000 quyết định, không thiếu mẫu/không có inference error được ghi nhận. Checkpoint riêng và ZIP tổng có prediction giống nhau từng byte. Không cần chạy lại hai model này để sửa hoặc bổ sung dữ liệu của lượt mặc định.

## Những kiểm tra đã hoàn thành

- Hash của exact `cases.jsonl` khớp hai manifest: `318f199b56cdc08d6db9865edbf73d8264017fe48dc909a550f7fde6aa54ee43`.
- Hai model cùng 3.400 case, gồm 400 workflow case × 5 câu hỏi và 3.000 OOD case × 1 câu hỏi.
- Đủ khóa `(domain, case_id, question)`, không trùng prediction; valid option/gold/probabilities; soft gold trong CSV khớp input snapshot. Không có text trùng sau chuẩn hóa lowercase/whitespace trong 3.400 case; đây không phải kiểm tra nhiễm dữ liệu training.
- Tái tính toàn bộ as-shipped metrics từ CSV: khớp bảng Colab tới sai số số thực. Code phân tích và inference được export khớp source đã bàn giao.
- Kiểm tra 20 seed/model: fit/eval không có case chung. Temperature giữ accuracy, kể cả hai trường hợp binary tie của d1-3B.
- d1-omni raw arm khớp phép đảo temperature explicit của cùng snapshot ở mọi dòng. Không dựng raw arm cho d1-3B.
- Tính thêm metrics theo từng task; 1.000 paired bootstrap/task (refit ID temperatures trong từng replicate); 20 calibration splits; paired model comparisons trên cùng test items.

Metadata Colab ghi GPU T4, FP16 theo source đã chạy. Vòng inference sau model loading mất khoảng 108 giây (omni) và 306 giây (3B); số này không bao gồm download/load và không phải benchmark latency kiểm soát.

## As-shipped trên OOD cân bằng

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

## Kết quả quan trọng nhất: refit trong workflow không đảm bảo transfer sang rating

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

## Raw/shipped ablation của d1-omni

Explicit shipped temperature đã giảm lỗi trên cả bốn OOD task trong lượt này. Riêng Yelp:

| Arm | ECE15 | NLL | Brier |
|---|---:|---:|---:|
| raw_reconstructed | 0.1203 | 0.8759 | 0.4910 |
| shipped | 0.0418 | 0.8108 | 0.4675 |
| ID_refit_per_type | 0.1155 | 0.8786 | 0.4930 |

Đây là ablation trực tiếp phép temperature trên cùng predictions/weights và cùng inputs, không chứng minh nguyên nhân training/architecture. Raw được khôi phục bằng phép đảo temperature đã kiểm tra với source, có giới hạn precision. T additional fit trên workflow nhân với T shipped; không được mô tả T additional như nhiệt độ tuyệt đối của logits gốc.

## Cách cập nhật paper

Luận điểm phù hợp: **calibration theo primitive vẫn phụ thuộc task; temperature sửa workflow có thể làm hỏng ordinal OOD vốn được calibrate tốt.** Đây là kết quả thực nghiệm cụ thể cho d1-omni và bộ Yelp được chọn, không phải định lý mới hoặc quy luật mọi decision model.

- Giữ bảng 5 checkpoint/3 release families từ corpus cũ làm exploratory cross-family context.
- Dùng hai d1 × 3.000 OOD mới làm extension có checkpoint hashes, exact input snapshot, class balance và ordinal metrics.
- Không gộp OOD 200 mẫu cũ với OOD 3.000 mới để khẳng định một xu hướng chung: câu chuyện transfer của omni đổi khi tập test lớn hơn và có score.
- Main finding mạnh hơn trên Yelp là cả ECE và NLL của omni đều xấu đi sau ID per-type refit, với percentile intervals dương ở split chính và 20/20 split xấu hơn.
- Workflow score gồm 700 quyết định 4 mức và 100 quyết định 5 mức; Yelp đều 5 mức. Shift thay cả semantics/rubric và phân bố số lựa chọn, nên chưa cô lập nguyên nhân do domain hay cardinality.
- Bootstrap là exploratory, không điều chỉnh multiple comparisons. Không dùng 20 seed như 20 dataset độc lập.

Có thể hoàn thiện bản thảo với các kết quả đã nhận. Nếu mở rộng claim semantic OOD **cross-family**, cần chạy ba checkpoint OpenDecider/Jeff trên đúng `cases_sha256`; hiện chúng chỉ có OOD nhỏ cũ. Không cần thêm inference chỉ để viết xong phần hai d1. Author/affiliation và venue vẫn cần chốt trước submission; teacher gold và novelty relative to prior work phải được trình bày đúng.

## Files và tái hiện

Bundle cập nhật giữ dữ liệu, source đã chạy, analysis theo task, bootstrap, paired model CI và bản thảo. `taskwise_metrics.csv` có ordinal RPS/MAE hard-target; dữ liệu typed có thêm soft-target metrics.

Chạy lại CPU analysis không dùng GPU. Dùng `verified/scripts/verify_analysis.py --models d1-omni d1-3b --data-dir colab_complete/extension_output`. Analysis bổ sung theo task có scripts riêng trong `checkpoint_omni/` và `colab_complete/`.
