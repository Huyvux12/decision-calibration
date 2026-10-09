# Kiểm chứng nghiên cứu decision calibration — 09/10/2026

**Kết luận:** có thể viết tiếp một bản thảo nghiên cứu thực nghiệm, nhưng báo cáo cũ chưa đủ tin cậy để dùng nguyên văn cho submission. Prediction hiện có dùng được cho phân tích lại; các kết luận về transfer, novelty và nguyên nhân calibration phải sửa. Bộ bàn giao có notebook Colab độc lập và mã nguồn để hoàn thành thực nghiệm mới.

## Những gì đã kiểm tra trực tiếp

- Đọc toàn bộ module dữ liệu, model adapter, metrics, temperature fitting và script phân tích.
- Kiểm tra 5 CSV × 2.200 quyết định = 11.000 phân phối; mỗi model có cùng các ID câu hỏi, không trùng khóa `(domain, case_id, question)`; probability hữu hạn, không âm, tổng gần 1; mọi gold label nằm trong option set.
- Tái tính accuracy, ECE, Brier, NLL từ CSV. Chuẩn hóa sai số tổng do lưu 6 chữ số thập phân trước khi phân tích mới. Sai số tổng tối đa trước chuẩn hóa khoảng 2×10⁻⁶.
- Smoke test cũ qua; thêm regression checks cho chia theo case, soft gold, tie handling và từ chối fallback temperature âm thầm.
- Lấy gold distribution của 400 test case từ nguồn typed-decisions để phân tích thêm soft-target KL, Brier, soft accuracy và ordinal RPS. Kiểm tra ID và hard label khớp với prediction. Bản gold và metadata nguồn được giữ trong `provenance/`.
- Phân tích mới: 20 cách chia tập độc lập theo seed, fit 20% case của từng workflow; 1.000 paired cluster bootstrap cho mỗi model × mode global/per-type × eval ID/OOD. Mỗi bootstrap lấy lại case ở cả fit và eval rồi **fit lại temperature**. Leave-one-workflow-out được tính cho cả global và per-type.

Đây là kiểm chứng **prediction và phân tích CPU**. Môi trường này không có PyTorch/GPU; chưa chạy inference mới trên Colab. Các run trước trong file đính kèm không có snapshot hash/runtime manifest đầy đủ, nên không thể xác nhận hồi tố chính xác checkpoint, thời gian GPU hoặc tuyên bố “0 warning”.

## Các lỗi quan trọng và cách sửa

| Vấn đề | Bằng chứng | Sửa trong bộ mới |
|---|---|---|
| Fit/eval chia theo từng decision của cùng một case | 266/400 case xuất hiện ở cả hai tập ở seed cũ; 400 decisions fit lấy từ 266 case | Chia theo `(workflow, case_id)` trước, rồi giữ tất cả 5 câu hỏi của case trong cùng tập: 80 fit case/320 eval case, giao bằng 0 |
| Per-task OOD “transfer” thực ra không hiệu chỉnh | Group key là `(domain,qtype)`; tên domain OOD không có trong temperature map, `.get(...,1.0)` luôn trả 1 | Không báo cáo per-task transfer khi task chưa được fit; thiếu group sẽ raise. Transfer chỉ dùng global hoặc per-type |
| Nhận định score khó nhất ở 4/5 model sai | ECE lớn nhất ở score chỉ với d1-omni và OpenDecider; 3 model còn lại choice lớn nhất | Bỏ kết luận phổ quát; kiểm tra từng primitive, từng workflow và OOD score riêng |
| Gọi 5 checkpoint từ “4 họ” | Hai Liquid, một OpenDecider, hai Jeff = 3 nhóm phát hành theo định nghĩa đang dùng | Viết “5 checkpoints across 3 release families”; không suy ra mỗi nhóm đồng nhất về kiến trúc |
| So sánh 0.251→0.038 của paper gốc không đúng phạm vi | Đây là Jev trên D3 social-science pooled; không phải một kết quả chung trên cùng ID/OOD của chúng ta | Không dùng để kết luận “d1 không replicate được” |
| Gọi protocol cũ là replicate trung thực | Paper gốc dùng 10 equal-mass bins; D1 fit per-type trên training states, D2/D3 5-fold item-disjoint; khác protocol 15 bins, test-split holdout 20% của bundle | Gọi là nghiên cứu mở rộng thực nghiệm; báo cả 5/10/15 mass bins và 10/15 width bins, không so con số trực tiếp giữa datasets |
| Gold bị mô tả như đúng/sai khách quan | typed-decisions là nhãn teacher tổng hợp, có full soft distribution | Gọi accuracy ID là teacher-reference agreement; tách hard-label calibration khỏi soft-distribution fidelity |
| “Cả hai d1 tốt nhờ per-type temperatures trong config” thiếu căn cứ | Snapshot hiện tại d1-omni có temperatures theo type×option-count; d1-3B không có field đó, engine được tạo không truyền calibration object | Chỉ xác nhận cơ chế explicit temperature của omni ở snapshot đã đọc; không quy nguyên nhân cho d1-3B hoặc snapshot cũ |
| Tie làm accuracy đổi sau scaling | 3/6/7 dòng noul của OpenDecider/Jeff-2B/Jeff-0.8B là p=0.5, stored pred chọn true; argmax theo labels sorted chọn false | Giữ lựa chọn argmax đã lưu khi tied, scaling giữ pred; test accuracy bất biến |
| Lưu probability 6 chữ số | Không có logits gốc/full precision của run cũ | Mô tả equivalence temperature là gần đúng do float precision; run mới lưu Python float precision |
| Inference cũ tiếp tục khi lỗi | `continue` có thể tạo CSV thiếu mẫu mà vẫn DONE | Resume theo khóa; manifest có expected/actual/missing/errors; run thiếu mẫu thất bại và không được đưa vào kết luận |
| Causal claim “calibration là thuộc tính pipeline” mạnh quá | Không có controlled ablation tách training/architecture/temperature | Định vị descriptive; thêm raw/shipped arm cho omni ở run mới; không kết luận nguyên nhân xuyên họ |

AG News cũ phân bố 60 nhãn: world 15, sports 11, business 5, scitech 29. Rotten Tomatoes thực sự 20/20; SST-2 thực sự 50/50. 200 OOD quyết định là kiểm tra nhỏ trên các task tương đối dễ, không chứng minh giữ calibration khi đổi domain nói chung.

## Cách đọc kết quả mới

`results_corrected/SUMMARY.md` và các CSV chứa số liệu được tái tính; chúng là nguồn cho bản thảo mới. `delta = after − before`: âm là tốt hơn cho NLL/Brier/ECE. Cần đọc NLL và Brier cùng ECE; ECE thấp không có nghĩa model chính xác hoặc phân biệt tốt. Uniform reference được xuất để minh họa.

- Các seed dùng lại phần lớn case, nên tỉ lệ seed cải thiện là độ ổn định, **không** là các thí nghiệm thống kê độc lập.
- CI bootstrap ở seed chính có fit lại T và giữ cấu trúc case. Không gọi một khác biệt có CI chứa 0 là kết quả chắc chắn; không dùng ECE để chọn mode tốt nhất sau khi xem test.
- ECE equal-mass có thể chia các confidence tied vào các bin khác nhau. Thử thêm equal-width; không đổi định nghĩa ECE mà không ghi rõ.
- Hard NLL là tiêu chí fit đã cố định. Soft KL/Brier là phân tích phụ về fidelity với teacher, không phải calibration đối với con người. RPS sử dụng thứ tự các mức; tên cột ghi rõ hard/soft target.
- Leave-one-workflow-out là shift trong cùng bộ benchmark, vẫn khác semantic OOD score Yelp.

## Kết luận cập nhật sau khi tính lại

- d1-omni **có lợi ích refit rõ hơn nhận định cũ**: qua 20 split theo case, ECE trung bình per-type 0.0790→0.0458, ΔNLL −0.0269. Bootstrap seed chính cho ΔNLL [−0.0410,−0.0031]; riêng ECE interval vẫn chứa 0.
- d1-3B gần như không thêm lợi ích NLL; ECE giảm nhỏ không đủ để nói mọi metric tốt hơn.
- Jeff-0.8B cải thiện ID mạnh nhất; OOD có thể giảm NLL nhưng tăng ECE, nên không gộp thành một verdict “transfer thất bại” đơn giản.
- OpenDecider per-type cải thiện ID NLL, nhưng OOD NLL xấu hơn trung bình +0.0359. Với OOD 200 mẫu, cần dùng bộ cân bằng lớn hơn để củng cố.
- Thực nghiệm nhiều seed là kiểm tra độ ổn định, không phải bằng chứng 20 lần độc lập. Các CI là exploratory và không điều chỉnh cho việc so nhiều metric/model.

## Thực nghiệm Colab cần chạy tiếp

Mặc định notebook chạy **d1-omni và d1-3B**, FP16, từng model ở subprocess riêng, trên 2.000 typed decisions được chạy lại và **3.000 OOD decisions**:

| Task OOD mới | Số mẫu mặc định | Cân bằng |
|---|---:|---|
| AG News test | 1.000 | 250/class × 4 |
| Rotten Tomatoes test | 500 | 250/class × 2 |
| SST-2 validation | 500 | 250/class × 2 |
| Yelp Review Full test, score 5 mức | 1.000 | 200/class × 5 |

Notebook đặt `QUOTA_PER_CLASS=250` cho ba task đầu và `YELP_PER_CLASS=200` cho rating. Quy mô cuối cùng luôn lấy từ manifest thực tế; lượt probe không được dùng làm kết quả paper.

Tất cả lựa chọn sample có seed; không đưa gold vào input; xuất JSONL chính xác input, IDs, schema và gold. Yelp label 0..4 được map sang Decision Index level 0..4 với rubric “1..5 sao”. Rating của người viết có yếu tố chủ quan; đây là phân loại ordinal theo text, không phải đánh giá một business khách quan. Model có thể đã thấy benchmark public trong training; không gọi đây là đảm bảo không nhiễm dữ liệu.

Notebook hỗ trợ thêm OpenDecider-small, Jeff-2B, Jeff-0.8B bằng đổi `MODELS_TO_RUN`. Cần cùng input hash và cùng resolved revisions khi ghép run nhiều session. Nếu có GPU lớn hơn, cũng có thể thử model khác, nhưng **không bắt buộc thêm family chỉ để tăng số lượng** trước khi xử lý các thiếu sót hiện tại.

Ưu tiên để hoàn thiện paper:

1. Chạy lại hai d1 bằng notebook mặc định; gửi ZIP output để cập nhật bảng và kiểm tra raw/shipped arm của omni.
2. Chạy cùng bộ OOD cho ba checkpoint còn lại nếu muốn giữ luận điểm cross-family về semantic OOD.
3. Ghi authors/affiliations và chọn venue; bổ sung kiểm tra nhãn thủ công hoặc ghi rõ giới hạn teacher gold; kiểm tra related work một lần nữa trước submission.

## Định vị paper

Đổi sang **“When Does Shipped Calibration Transfer? A Case-Disjoint Study of Open Decision Models”**. Novelty nên nằm ở các checkpoint d1 mới, phân tích có tách case/primitive, độ ổn định transfer và ordinal OOD. Temperature scaling và nhận định calibration phụ thuộc task đã có trong prior work; không tuyên bố đây là thuật toán mới hoặc nghiên cứu đầu tiên nếu chưa có rà soát hệ thống. Không có bảo đảm chấp nhận paper từ các con số hiện tại.

## Nguồn đã đối chiếu

- [Rafe & Das, arXiv:2610.00346v2](https://arxiv.org/html/2610.00346v2), §3.2–3.3, Table 7 và RQ2.
- [typed-decisions dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), gold/distributions và hướng dẫn diễn giải điểm.
- [d1-omni-600M model card](https://huggingface.co/LiquidAI/d1-omni-600M); source/config snapshot `02b55d7076f15129e59ab3f94783f32c4b088674`.
- [d1-3B model card](https://huggingface.co/LiquidAI/d1-3B); source/config snapshot `051bcc464b01b9f92942b364d9586b0ef5912432`.
- [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full).
- [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html), temperature scaling.

`REPORT.md`, `PAPER_OUTLINE.md`, README cũ được giữ dưới tên `LEGACY_*` chỉ để đối chiếu. Không dùng các câu đã bị đánh dấu sai ở trên làm evidence trong paper.
