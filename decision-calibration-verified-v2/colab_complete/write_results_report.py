import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
read=lambda p:list(csv.DictReader(p.open()))
M0=read(ROOT.parent/'checkpoint_omni/taskwise/taskwise_metrics.csv')
M1=read(ROOT/'taskwise_d1-3b/taskwise_metrics.csv')
R0=read(ROOT.parent/'checkpoint_omni/taskwise/taskwise_repeated_splits.csv')
R1=read(ROOT/'taskwise_d1-3b/taskwise_repeated_splits.csv')
C0=read(ROOT.parent/'checkpoint_omni/taskwise/taskwise_bootstrap.csv')
C1=read(ROOT/'taskwise_d1-3b/taskwise_bootstrap.csv')
PAIRED=read(ROOT/'paired_model_differences.csv')
DOMAINS=['ood_agnews_balanced','ood_rotten_balanced_v2','ood_sst2_balanced_v2','ood_yelp_rating']
NAMES=['AG News','Rotten Tomatoes','SST-2','Yelp rating']
models=[('d1-omni',M0,R0,C0),('d1-3b',M1,R1,C1)]
s='''# Kết quả Colab đã xác minh — hai checkpoint d1

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
'''
for m,metrics,repeated,cis in models:
 for d,n in zip(DOMAINS,NAMES):
  r=next(x for x in metrics if x['domain']==d and x['qtype']=='all' and x['arm']=='shipped')
  s+='| '+' | '.join([m,n,r['n']]+[f"{float(r[k]):.4f}" for k in ['accuracy','mean_confidence','ece_mass_15','nll','brier']])+' |\n'
s+='''
Yelp cân bằng 200 review/mỗi mức, 5 mức. Level 0..4 tương ứng 1..5 sao; không cộng hoặc trừ nhãn ngoài mapping này. Hai model có accuracy Yelp gần nhau; không suy ra 3B tốt hơn trên rating chỉ từ số tham số. Paired model differences và CI được cung cấp trong bundle.

## Kết quả quan trọng nhất: refit trong workflow không đảm bảo transfer sang rating

Bảng sau là trung bình qua 20 calibration splits theo case. “Per-type” dùng riêng T fit trên workflow score để áp cho Yelp score; accuracy giữ nguyên.

| Model | Mode | Yelp ECE shipped→refit | Yelp NLL shipped→refit | Số split NLL xấu hơn |
|---|---|---:|---:|---:|
'''
import numpy as np
for m,metrics,rr,cis in models:
 for mode in ['global','per_type']:
  r=[x for x in rr if x['domain']=='ood_yelp_rating' and x['mode']==mode];avg=lambda key:float(np.mean([float(x[key]) for x in r]))
  s+=f"| {m} | {mode} | {avg('before_ece_mass_15'):.4f}→{avg('after_ece_mass_15'):.4f} | {avg('before_nll'):.4f}→{avg('after_nll'):.4f} | {sum(float(x['delta_nll'])>0 for x in r)}/20 |\n"
s+='''
Với **d1-omni**, workflow refit giúp ID nhưng làm model thiếu tự tin trên Yelp vốn có mean confidence gần đúng accuracy. Với **d1-3B**, ECE và NLL có thể đi theo hai hướng khác nhau; không gọi giảm ECE là cải thiện mọi khía cạnh của phân phối xác suất.

Các interval dưới đây dùng seed chính 20261009, nên point delta khác trung bình 20 seed. Delta âm là giảm loss/ECE. Bootstrap có refit temperature, không chỉ lấy lại test samples.

| Model | Yelp comparison | Metric | Δ | 95% percentile interval |
|---|---|---|---:|---|
'''
for m,metrics,rr,cis in models:
 for r in cis:
  if r['domain']=='ood_yelp_rating' and r['comparison']=='ID_per_type_minus_shipped' and r['metric'] in ['ece_mass_15','nll']:
   s+=f"| {m} | per-type refit − shipped | {r['metric']} | {float(r['delta']):.4f} | [{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}] |\n"
s+='''
## Raw/shipped ablation của d1-omni

Explicit shipped temperature đã giảm lỗi trên cả bốn OOD task trong lượt này. Riêng Yelp:

| Arm | ECE15 | NLL | Brier |
|---|---:|---:|---:|
'''
for arm in ['raw_reconstructed','shipped','ID_refit_per_type']:
 r=next(x for x in M0 if x['domain']=='ood_yelp_rating' and x['qtype']=='all' and x['arm']==arm);s+=f"| {arm} | {float(r['ece_mass_15']):.4f} | {float(r['nll']):.4f} | {float(r['brier']):.4f} |\n"
s+='''
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
'''
(ROOT/'COLAB_RESULTS_VERIFIED.md').write_text(s)
print(ROOT/'COLAB_RESULTS_VERIFIED.md')
