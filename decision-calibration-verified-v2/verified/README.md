# Decision calibration — verified continuation package

Bắt đầu bằng `../COLAB_RESULTS_VERIFIED.md` cho kết quả Colab hoàn tất. `VERIFY_REPORT.md` là audit corpus lịch sử; bản thảo tiếng Anh là `PAPER_DRAFT.md`. `results_corrected/SUMMARY.md` chỉ tóm tắt corpus lịch sử. Các lệnh dưới đây chạy từ thư mục `verified/`.

## Chạy Colab

1. Tải `Decision_Calibration_Verified_Colab.ipynb`, vào https://colab.research.google.com → File → Upload notebook.
2. Runtime → Change runtime type → GPU. T4 là cấu hình tối thiểu dự kiến cho hai d1; L4 phù hợp nếu có Colab Pro. Không bật BF16 cho d1-omni; script dùng FP16.
3. Giữ mặc định và Run all. Notebook tự giải nén code/dữ liệu nhúng, phân tích lại prediction cũ, chạy mới `d1-omni`, `d1-3b`, rồi xuất `decision_calibration_outputs.zip`.
4. Gửi **ZIP output**, không chỉ ảnh bảng kết quả. Nếu run lỗi, chạy cell export cuối để gửi log/manifest; giữ output folder và chạy lại để tiếp tục case còn thiếu.

Không cần tải ZIP nguồn lên Colab; notebook đã nhúng code, 5 prediction CSV và gold snapshot. Không cần nhập API key cho các public checkpoints. Nếu HF yêu cầu login về sau thì dùng login của HF trong Colab, không ghi token vào file kết quả.

Các biến chính trong cell cấu hình:

```python
RUN_GPU_EXTENSION = True
MODELS_TO_RUN = ['d1-omni', 'd1-3b']
QUOTA_PER_CLASS = 250
YELP_PER_CLASS = 200
N_SPLIT_SEEDS = 20
N_BOOTSTRAP = 1000
USE_DRIVE = False
```

Mặc định mỗi model mới: 400 typed case/2.000 decisions + 3.000 OOD decisions = 5.000 rows. AG News 1.000, Rotten 500, SST-2 500, Yelp 1.000. Yelp level 0..4 ứng với 1..5 sao. Để chạy thử nhanh, đặt quota=25, Yelp=20; không trộn lượt thử vào bảng chính. Lượt probe chỉ kiểm tra API và schema.

Nếu muốn semantic OOD **cross-family**, đổi:

```python
MODELS_TO_RUN = ['d1-omni', 'd1-3b', 'opendecider-small', 'jeff-2b', 'jeff-0.8b']
```

Có thể chia nhiều session bằng giữ output trên Drive (`USE_DRIVE=True`) và giữ nguyên folder/config; tất cả phải dùng cùng `resolved_revisions.json` và `cases_sha256`. Các model được chạy từng subprocess, nên giải phóng VRAM sau mỗi run. Không thay model ở cùng một kernel đã load toàn bộ model.

Notebook xuất phân tích CPU hiện có, CSV inference full precision, source input JSONL, resolved hashes, pip versions, manifest completeness, log và plots. `actual_rows` phải bằng `expected_rows`, `complete=True`. Nhánh raw cho d1-omni khôi phục từ chính nhiệt độ explicit của snapshot mới; không suy ngược nhiệt độ lịch sử từ snapshot hiện tại.

## Chạy bằng terminal

Phân tích từ prediction cũ, không cần GPU:

```bash
pip install numpy scipy matplotlib
python scripts/test_verified.py
python scripts/verify_analysis.py --seeds 20 --bootstrap 1000 --out results_corrected
python scripts/summarize_verified.py results_corrected
python scripts/make_verified_figures.py results_corrected
```

Inference mới trên GPU đã cài dependencies:

```bash
python scripts/run_extension.py --model d1-omni --include-typed --quota 250 --yelp-per-class 200 --out extension_output --probe
python scripts/run_extension.py --model d1-omni --include-typed --quota 250 --yelp-per-class 200 --out extension_output
python scripts/run_extension.py --model d1-3b --include-typed --quota 250 --yelp-per-class 200 --out extension_output
python scripts/verify_analysis.py --data-dir extension_output --models d1-omni d1-3b --out extension_results
python scripts/summarize_verified.py extension_results
```

Dùng calibration global/per-type để transfer sang task chưa biết. Per-task chỉ được đánh giá với task đã có calibration data; script từ chối group không tồn tại. Delta âm là cải thiện cho loss/ECE. Seed spread không phải CI. Ordinal RPS có phân biệt hard/soft target.

## Trạng thái xác minh

CPU: regression checks và phân tích 20 seeds/1.000 bootstrap trên 11.000 historical distributions. GPU extension: người dùng đã trả đủ 5.000 prediction/model cho d1-omni và d1-3B. Đã kiểm tra input hashes, manifests, source, toàn bộ rows/gold/probabilities, tái tính metrics và bổ sung analysis theo từng task. Hai model không cần chạy lại cho lượt mặc định. Kết quả mới ở `../colab_complete/`; omni taskwise analysis ở `../checkpoint_omni/`.

File `LEGACY_*`, `scripts/run_rq2.py`, `dist/` và figures cũ là lịch sử. Script analysis mới có import vài helper định nghĩa từ `run_rq2.py` nhưng **không** chạy protocol cũ làm kết quả cuối. Sửa requirements cũ: bỏ dấu quote quanh requirement transformers. Không dùng các claims bị chỉ ra trong verification report.
