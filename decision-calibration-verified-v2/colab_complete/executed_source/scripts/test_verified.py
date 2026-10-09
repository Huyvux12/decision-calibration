"""Regression checks for the consequential protocol fixes; CPU only."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import verify_analysis as V
rows,_=V.load(V.ROOT/'data/predictions_d1-omni.csv');ind=[r for r in rows if r['domain'].startswith('typed_')]
for seed in range(20):
 a,b=V.split_cases(ind,.2,seed);assert len(a)==400 and len(b)==1600;assert not {V.key(r) for r in a}&{V.key(r) for r in b}
for model in V.old.DEFAULT_MODELS:
 r,issues=V.load(V.ROOT/'data'/('predictions_'+model+'.csv'));p=V.scale(r,{'all':2.5},'global')
 assert [x['pred_idx'] for x in r]==[x['pred_idx'] for x in p]
 assert V.stats(r)['accuracy']==V.stats(p)['accuracy']
 assert all(abs(x['probs'].sum()-1)<1e-12 for x in p)
 assert len(r)==2200
 # Recompute supplied as-shipped metrics with their exact original definitions.
 for dg in ['in-domain','ood']:
  for qt in ['noul','choice','score']:
   rr=[x for x in r if V.old.domain_group(x['domain'])==dg and x['qtype']==qt]
   if not rr:continue
   old=V.old.M.summarize(rr);new=V.stats(rr)
   assert abs(old['accuracy']-new['accuracy'])<1e-12
   assert abs(old['ece']-new['ece_mass_15'])<1e-12
 assert V.hydrate(r)==2000
try:V.scale(rows,{('typed_customer_service','noul'):2},'per_task')
except ValueError:pass
else:raise AssertionError('Unknown group must not silently use T=1')
# Soft-label and hard-label scoring must stay separately named.
V.hydrate(rows);s=V.stats(rows);assert 'soft_kl' in s and 'nll' in s and s['soft_n']==2000
print('PASS: 20 case splits disjoint; 11,000 distributions; ties preserved; metric equivalence; soft gold 10,000 decisions; missing group rejected.')
