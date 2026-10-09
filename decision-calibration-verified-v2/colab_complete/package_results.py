"""Build the completed, auditable research handoff; run from workspace root."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT/'deliverables' if (ROOT/'deliverables').is_dir() else ROOT
DEST = REPORTS / 'decision-calibration-verified-v2.zip'
README = '''# Decision calibration — completed Colab handoff

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
'''


def collect(directory, prefix, excluded=()):
    for p in sorted(directory.rglob('*')):
        rel = p.relative_to(directory)
        if not p.is_file() or '__pycache__' in rel.parts or p.suffix == '.pyc':
            continue
        if any(rel == Path(x) or Path(x) in rel.parents for x in excluded):
            continue
        yield str(Path(prefix) / rel), p.read_bytes()


def main():
    files = {'README.md': README.encode()}
    for name in ['REPORT_FINAL.md', 'PAPER_DRAFT.md', 'VERIFY_REPORT.md', 'COLAB_RESULTS_VERIFIED.md',
                 'Decision_Calibration_Verified_Colab.ipynb']:
        files[name] = (REPORTS / name).read_bytes()
    files.update(collect(ROOT / 'verified', 'verified', excluded=[
        'results_final', 'results_paper', 'results_verified', 'dist',
        'provenance/source', 'PACKAGE_MANIFEST.json']))
    files.update(collect(ROOT / 'verified/results_corrected', 'results_corrected'))
    files.update(collect(ROOT / 'colab_complete', 'colab_complete'))
    files.update(collect(ROOT / 'checkpoint_omni', 'checkpoint_omni'))
    manifest = {
        'description': 'Completed two-d1 Colab export and verified CPU analyses',
        'cases_sha256': '318f199b56cdc08d6db9865edbf73d8264017fe48dc909a550f7fde6aa54ee43',
        'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                  for name, data in sorted(files.items())},
    }
    files['PACKAGE_MANIFEST.json'] = json.dumps(manifest, indent=2).encode()
    with zipfile.ZipFile(DEST, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(files.items()):
            z.writestr(name, data)
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        for name, metadata in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == metadata['sha256']
        assert z.read('verified/PAPER_DRAFT.md') == z.read('PAPER_DRAFT.md')
        assert z.read('colab_complete/extension_output/predictions_d1-omni.csv') == z.read('checkpoint_omni/extension_output/predictions_d1-omni.csv')
        for name in ['results_corrected/figures/calibration_delta_ci.png',
                     'colab_complete/taskwise_temperature_effects.png']:
            assert name in z.namelist()
    print(json.dumps({'archive': str(DEST), 'files': len(files),
                      'bytes': DEST.stat().st_size, 'crc_and_sha256': 'PASS'}))


if __name__ == '__main__':
    main()
