"""Generate readable summaries of result CSVs, including ordinal OOD slices."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def read(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder');a=ap.parse_args();p=Path(a.folder)
 rows=read(p/'repeated_group_split.csv');base=read(p/'as_shipped_verified.csv');ci=read(p/'paired_cluster_bootstrap.csv');loo=read(p/'leave_one_workflow_out.csv')
 lines=['# Verified calibration results','', 'Delta = after − before. Negative losses/ECE indicate improvement. Repeated splits are dependent stability measurements; their spread is not a confidence interval. Bootstrap intervals resample cases and refit temperatures.','', '## As shipped: whole datasets','', '| Model | Dataset | n | Accuracy | ECE mass15 | NLL | Brier |','|---|---|---:|---:|---:|---:|---:|']
 for r in base:
  if r['qtype']=='all':lines.append('| '+' | '.join([r['model'],r['domain'],r['n']]+[f"{float(r[k]):.4f}" for k in ['accuracy','ece_mass_15','nll','brier']])+' |')
 lines+=['','## Repeated case-disjoint calibration splits (all types pooled)','','| Model | Eval | Mode | Splits | Mean ECE before→after | Mean ΔNLL | ECE improves in |','|---|---|---|---:|---:|---:|---:|']
 groups={}
 for r in rows:
  if r['qtype']=='all':groups.setdefault((r['model'],r['eval'],r['mode']),[]).append(r)
 summary=[]
 for (m,e,mode),rr in sorted(groups.items()):
  mean=lambda key:float(np.mean([float(r[key]) for r in rr]));improve=sum(float(r['delta_ece_mass_15'])<0 for r in rr)
  lines.append(f"| {m} | {e} | {mode} | {len(rr)} | {mean('before_ece_mass_15'):.4f}→{mean('after_ece_mass_15'):.4f} | {mean('delta_nll'):.4f} | {improve}/{len(rr)} |")
  summary.append(dict(model=m,eval=e,mode=mode,n_seeds=len(rr),ece_before_mean=mean('before_ece_mass_15'),ece_after_mean=mean('after_ece_mass_15'),delta_nll_mean=mean('delta_nll'),ece_improves_splits=improve))
 lines+=['','## Paired cluster bootstrap (seed 0; fit refitted each replicate)','','| Model | Eval | Mode | Metric | Δ | 95% percentile interval |','|---|---|---|---|---:|---|']
 for r in ci:
  if r['metric'] in ['nll','ece_mass_15']:lines.append(f"| {r['model']} | {r['eval']} | {r['mode']} | {r['metric']} | {float(r['delta']):.4f} | [{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}] |")
 lines+=['','## Leave-one-workflow-out (per-type, score questions only)','','| Model | Held workflow | ECE before→after | NLL before→after |','|---|---|---:|---:|']
 for r in loo:
  if r['mode']=='per_type' and r['qtype']=='score':lines.append(f"| {r['model']} | {r['held_workflow']} | {float(r['before_ece_mass_15']):.4f}→{float(r['after_ece_mass_15']):.4f} | {float(r['before_nll']):.4f}→{float(r['after_nll']):.4f} |")
 (p/'SUMMARY.md').write_text('\n'.join(lines)+'\n');(p/'repeated_summary.json').write_text(json.dumps(summary,indent=2));print(p/'SUMMARY.md')
if __name__=='__main__':main()
