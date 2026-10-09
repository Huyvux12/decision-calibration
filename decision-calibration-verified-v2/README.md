# Decision calibration — completed Colab handoff

Hai model d1 đều đủ 5.000 predictions; bắt đầu bằng REPORT_FINAL.md để đọc
báo cáo đầy đủ gồm phương pháp, audit, kết quả, giới hạn và evidence→code.
COLAB_RESULTS_VERIFIED.md là report ngắn riêng cho hai runs mới.
PAPER_DRAFT.md là bản thảo tiếng Anh đã cập nhật. VERIFY_REPORT.md ghi audit
ban đầu trên corpus lịch sử, trước các runs mới. Notebook vẫn là notebook
đã chạy thành công; không cần chạy lại mặc định.

## Nội dung

- verified/: source/data/provenance và kết quả CPU của 5 checkpoint lịch sử.
- results_corrected/: cùng kết quả lịch sử, đặt ở root để hình trong paper hoạt động.
- colab_complete/: export mới của cả hai d1; exact cases, predictions,
  checkpoint hashes, source đã chạy, manifests/logs, dependency versions,
  pooled metrics, taskwise 3B, paired model intervals và figure PNG/PDF.
- checkpoint_omni/: checkpoint omni được đối chiếu với export tổng;
  analysis theo task, raw/shipped ablation và refitting-bootstrap.
- PACKAGE_MANIFEST.json: SHA256/size của mọi file khác trong archive.

## Tái hiện CPU từ thư mục giải nén

```bash
python -m pip install numpy scipy matplotlib
python verified/scripts/test_verified.py
python verified/scripts/test_extension_io.py
python verified/scripts/verify_analysis.py --data-dir colab_complete/extension_output --models d1-omni d1-3b --seeds 20 --bootstrap 1000 --out reproduced_extension
python verified/scripts/summarize_verified.py reproduced_extension
python checkpoint_omni/analyze_checkpoint.py --bootstrap 1000
python colab_complete/analyze_tasks.py --bootstrap 1000
python colab_complete/paired_models.py
python colab_complete/make_task_figures.py
python colab_complete/write_results_report.py
```

Các script taskwise ghi lại output tương ứng; nên dùng một bản giải nén riêng
nếu muốn giữ nguyên bảng bàn giao. Không cần tải model hoặc GPU cho CPU analysis.
`colab_complete/executed_source/` giữ source đúng lượt đã chạy; source hiện tại
trong `verified/` có thêm cập nhật manuscript/README và chú thích dependencies.

Main finding: omni workflow refit làm Yelp ECE và NLL xấu hơn; 3B không có
cùng tác động NLL nhất quán. Bảng 20 split là stability summary, không phải
20 dataset độc lập. Bootstrap refit T và resample theo case, 1.000 replicate;
intervals mang tính exploratory và chưa điều chỉnh multiple comparisons.
Phần OOD mới chỉ có hai d1; OpenDecider/Jeff giữ corpus OOD nhỏ lịch sử.
