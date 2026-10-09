"""Audit submitted checkpoint, taskwise raw/shipped/ID-refitted probability quality."""
import argparse,csv,hashlib,json,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'verified/scripts'));import verify_analysis as V

def case_resample_indices(rows,rng):
    blocks=defaultdict(list)
    for i,r in enumerate(rows):blocks[V.key(r)].append(i)
    blocks=list(blocks.values());return [j for i in rng.integers(0,len(blocks),len(blocks)) for j in blocks[i]]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=1000);args=ap.parse_args()
    p=ROOT/'extension_output';out=ROOT/'taskwise_d1-3b';out.mkdir(exist_ok=True)
    cases=[json.loads(l) for l in (p/'cases.jsonl').open()];rows,issues=V.load(p/'predictions_d1-3b.csv');raw=rows;rawissues={};m=json.loads((p/'manifest_d1-3b.json').read_text())
    assert hashlib.sha256((p/'cases.jsonl').read_bytes()).hexdigest()==m['cases_sha256']
    expected={(c['domain'],c['row_id'],q) for c in cases for q in c['gold']};assert expected=={V.qkey(r) for r in rows}=={V.qkey(r) for r in raw};assert len(rows)==m['actual_rows']==m['expected_rows']==5000
    cm={(c['domain'],c['row_id']):c for c in cases};statehashes=defaultdict(list);state_byscope=defaultdict(set)
    for c in cases:statehashes[' '.join(c['state'].lower().split())].append(V.key(c))
    assert len(cm)==len(cases)
    for r,rr in zip(rows,raw):
        assert V.qkey(r)==V.qkey(rr) and r['labels']==rr['labels'] and r['gold_idx']==rr['gold_idx'] and r['pred_idx']==rr['pred_idx']
        c=cm[V.key(r)];g=c['gold'][r['question']];assert r['qtype']==g['type'] and r['labels'][r['gold_idx']]==str(g['label'])
        if 'probabilities' in g:assert np.allclose(r['gold_probs'],[g['probabilities'][l] for l in r['labels']])
    old,_=V.load(ROOT.parent/'verified/data/predictions_d1-3b.csv');oldmap={V.qkey(r):r for r in old};ind=[r for r in rows if r['domain'].startswith('typed_')]
    overlap={V.qkey(r) for r in ind}&set(oldmap)
    changes=sum(r['pred_idx']!=oldmap[V.qkey(r)]['pred_idx'] for r in ind)
    audit=dict(row_count=len(rows),case_count=len(cases),cases_sha256=m['cases_sha256'],complete=True,gold_alignment=True,raw_reconstruction_verified=False,normalized_exact_text_duplicates=[v for v in statehashes.values() if len(v)>1],old_new_typed_selection_changes=changes,old_new_probability_max_difference=float(max(abs(r['probs']-oldmap[V.qkey(r)]['probs']).max() for r in ind)),probability_validation_issues=issues,revision=m['revision'],bootstrap=args.bootstrap)
    (out/'audit.json').write_text(json.dumps(audit,indent=2));fit,eval=V.split_cases(ind,.2,20261009);temps={mode:V.fit(fit,mode) for mode in ['global','per_type']}
    comparisons=[];intervals=[];repeat=[]
    for domain in sorted({r['domain'] for r in rows}):
        rr=[r for r in rows if r['domain']==domain];r0=[r for r in raw if r['domain']==domain]
        scopes=[('all',rr,r0)]+[(qt,[r for r in rr if r['qtype']==qt],[r for r in r0 if r['qtype']==qt]) for qt in sorted({r['qtype'] for r in rr})]
        for qt,rs,rs0 in scopes:
            bs=V.stats(rs);raws=V.stats(rs0)
            comparisons.append(dict(domain=domain,qtype=qt,arm='shipped',**bs));# No raw reconstruction for d1-3B
            if not domain.startswith('typed_'):
                for mode in ['global','per_type']:comparisons.append(dict(domain=domain,qtype=qt,arm='ID_refit_'+mode,**V.stats(V.scale(rs,temps[mode],mode))))
        if domain.startswith('typed_'):continue
        bs=V.stats(rr);rs=V.stats(r0)
        for mode in ['global','per_type']:
            ps=V.stats(V.scale(rr,temps[mode],mode));diff=defaultdict(list);rng=np.random.default_rng(20261009)
            for b in range(args.bootstrap):
                aa=V.bootstrap_cases(fit,rng);ii=case_resample_indices(rr,rng);bb=[rr[i] for i in ii];s=V.stats(bb,full=False);p2=V.stats(V.scale(bb,V.fit(aa,mode),mode),full=False)
                for met in ['ece_mass_15','nll','brier']:diff[met].append(p2[met]-s[met])
            for met,v in diff.items():
                lo,hi=np.quantile(v,[.025,.975]);intervals.append(dict(domain=domain,comparison='ID_'+mode+'_minus_shipped',metric=met,delta=ps[met]-bs[met],ci_low=lo,ci_high=hi,B=args.bootstrap))
            for seed in range(20):
                a,b=V.split_cases(ind,.2,20261009+seed);ss=V.stats(V.scale(rr,V.fit(a,mode),mode));repeat.append(dict(domain=domain,mode=mode,seed=seed,**{'before_'+k:v for k,v in bs.items()},**{'after_'+k:v for k,v in ss.items()},**{'delta_'+k:ss[k]-v for k,v in bs.items() if k in ss and k!='n'}))
        print(domain,'done',flush=True)
    V.write_csv(out/'taskwise_metrics.csv',comparisons);V.write_csv(out/'taskwise_bootstrap.csv',intervals);V.write_csv(out/'taskwise_repeated_splits.csv',repeat)
    (out/'fit_temperatures_seed0.json').write_text(json.dumps(temps,indent=2))
    print('PASS full checkpoint audit;',out,flush=True)
if __name__=='__main__':main()
