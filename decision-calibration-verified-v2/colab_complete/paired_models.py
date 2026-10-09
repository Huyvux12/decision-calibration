import csv,sys,json
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'verified/scripts'));import verify_analysis as V
p=ROOT/'extension_output';a,_=V.load(p/'predictions_d1-omni.csv');b,_=V.load(p/'predictions_d1-3b.csv');assert [V.qkey(r) for r in a]==[V.qkey(r) for r in b]
out=[]
for d in sorted({r['domain'] for r in a if not r['domain'].startswith('typed_')}):
 ar=[r for r in a if r['domain']==d];br=[r for r in b if r['domain']==d];aa=V.stats(ar);bb=V.stats(br);rng=np.random.default_rng(20261009);boot=defaultdict(list)
 for n in range(1000):
  ix=rng.integers(0,len(ar),len(ar));ascore=V.stats([ar[i] for i in ix],full=False);bscore=V.stats([br[i] for i in ix],full=False)
  for k in ['accuracy','ece_mass_15','nll','brier']:boot[k].append(bscore[k]-ascore[k])
 for k,v in boot.items():
  low,high=np.quantile(v,[.025,.975]);out.append(dict(domain=d,metric=k,direction='d1-3b minus d1-omni',delta=bb[k]-aa[k],ci_low=low,ci_high=high,B=1000))
V.write_csv(ROOT/'paired_model_differences.csv',out);print('Saved paired model comparisons')
