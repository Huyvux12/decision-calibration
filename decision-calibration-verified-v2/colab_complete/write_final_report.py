"""Assemble the Vietnamese handoff report from verified reports and CSV summaries."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent


def nested(text):
    return re.sub(r'(?m)^(#{1,5}) ', r'\1# ', text)


intro = r'''# Báo cáo nghiên cứu hoàn chỉnh và bàn giao minh bạch

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

'''

tail = r'''

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
'''


def main():
    completed = (ROOT/'colab_complete/COLAB_RESULTS_VERIFIED.md').read_text()
    historical = (ROOT/'verified/results_corrected/SUMMARY.md').read_text()
    report = intro + nested(completed) + '\n\n## 6. Kết quả corpus lịch sử — phân tích lại\n\n'
    report += 'Toàn bộ bảng dưới đây chỉ dùng corpus lịch sử: OOD n=200/model. Không diễn giải chúng như kết quả trên 3.000 OOD items mới.\n\n'
    report += nested(historical) + tail
    reports = ROOT/'deliverables' if (ROOT/'deliverables').is_dir() else ROOT
    dest = reports/'REPORT_FINAL.md'
    dest.write_text(report)
    print(f'Saved {dest}; {len(report.encode())} bytes')


if __name__ == '__main__':
    main()
