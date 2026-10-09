import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import verify_analysis as V
ap=argparse.ArgumentParser();ap.add_argument('output_dir');a=ap.parse_args();p=Path(a.output_dir)
s,_=V.load(p/'predictions_d1-omni.csv');r,_=V.load(p/'raw_predictions_d1-omni.csv');assert [V.qkey(x) for x in s]==[V.qkey(x) for x in r]
out=[]
for d in sorted({x['domain'] for x in s}):
 for qt in sorted({x['qtype'] for x in s if x['domain']==d}):
  a=[x for x in s if x['domain']==d and x['qtype']==qt];b=[x for x in r if x['domain']==d and x['qtype']==qt];ss=V.stats(a);rr=V.stats(b)
  out.append(dict(domain=d,qtype=qt,**{'shipped_'+k:v for k,v in ss.items()},**{'raw_'+k:v for k,v in rr.items()}))
V.write_csv(p/'raw_vs_shipped_omni.csv',out);print(p/'raw_vs_shipped_omni.csv')
