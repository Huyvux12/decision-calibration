"""Case-disjoint calibration audit. Runs without a GPU or model downloads."""
import argparse, csv, hashlib, json, platform, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_rq2 as old

METRICS=['accuracy','mean_confidence','nll','brier','ece_mass_5','ece_mass_10','ece_mass_15','ece_width_10','ece_width_15','signed_gap']
def key(r): return (r['domain'],r['row_id'])
def qkey(r): return (*key(r),r['question'])
def pack(rows):
    k=max(len(r['probs']) for r in rows)
    p=np.zeros((len(rows),k)); g=np.array([r['gold_idx'] for r in rows]);
    for i,r in enumerate(rows): p[i,:len(r['probs'])]=r['probs']
    mask=np.arange(k)[None,:]<np.array([len(r['probs']) for r in rows])[:,None]
    return p,g,mask

def scale(rows, temps, mode):
    p,g,mask=pack(rows); ts=[]
    for r in rows:
        group='all' if mode=='global' else r['qtype'] if mode=='per_type' else (r['domain'],r['qtype'])
        if group not in temps: raise ValueError('Unsupported transfer group: '+str(group))
        ts.append(temps[group])
    lp=np.where(mask,np.log(np.maximum(p,1e-12)),-np.inf)/np.array(ts)[:,None]
    pr=np.exp(lp-logsumexp(lp,axis=1,keepdims=True));out=[]
    for i,r in enumerate(rows):
        nr=dict(r);nr['probs']=pr[i,:len(r['probs'])];nr['pred_idx']=r['pred_idx'];out.append(nr)
    return out

def fit(rows,mode):
    grouped=defaultdict(list)
    for r in rows:
        group='all' if mode=='global' else r['qtype'] if mode=='per_type' else (r['domain'],r['qtype'])
        grouped[group].append(r)
    out={}
    for group,rr in grouped.items():
        p,g,mask=pack(rr);lp=np.where(mask,np.log(np.maximum(p,1e-12)),-np.inf)
        def loss(logt):
            z=lp/np.exp(logt);return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(g)),g]))
        res=minimize_scalar(loss,bounds=(np.log(.05),np.log(10)),method='bounded',options={'xatol':1e-7})
        if not res.success: raise RuntimeError('Temperature fit failed')
        out[group]=float(np.exp(res.x))
    return out

def split_cases(rows,frac,seed):
    by=defaultdict(set)
    for r in rows:by[r['domain']].add(key(r))
    rng=np.random.default_rng(seed);selected=set()
    for d in sorted(by):
        keys=sorted(by[d]); ix=rng.permutation(len(keys));n=max(1,min(len(keys)-1,round(frac*len(keys))))
        selected.update(keys[i] for i in ix[:n])
    a=[r for r in rows if key(r) in selected];b=[r for r in rows if key(r) not in selected]
    assert not {key(r) for r in a}&{key(r) for r in b}
    return a,b

def stats(rows,full=True):
    p,g,_=pack(rows);pred=np.array([r['pred_idx'] for r in rows]); c=p[np.arange(len(rows)),pred]; y=(pred==g).astype(float)
    result={'n':len(rows),'accuracy':float(y.mean()),'mean_confidence':float(c.mean()),'nll':float(-np.log(np.maximum(p[np.arange(len(rows)),g],1e-12)).mean()),'brier':float((np.sum(p*p,axis=1)-2*p[np.arange(len(rows)),g]+1).mean()),'signed_gap':float(c.mean()-y.mean())}
    order=np.argsort(c,kind='stable')
    for nb in (5,10,15):
        result['ece_mass_'+str(nb)]=float(sum(abs(y[ix].mean()-c[ix].mean())*len(ix)/len(y) for ix in np.split(order,np.linspace(0,len(order),nb+1).astype(int)[1:-1]) if len(ix)))
    for nb in (10,15):
        ids=np.minimum((c*nb).astype(int),nb-1)
        result['ece_width_'+str(nb)]=float(sum(abs(y[ids==b].mean()-c[ids==b].mean())*(ids==b).mean() for b in range(nb) if np.any(ids==b)))
    if not full:return result
    soft=[i for i,r in enumerate(rows) if 'gold_probs' in r]
    if soft:
        kl=[]; bs=[]; sa=[]; rps=[]; mae=[]
        for i in soft:
            r=rows[i];s=np.array(r['gold_probs']);pr=r['probs'];kl.append(np.sum(np.where(s>0,s*(np.log(np.maximum(s,1e-12))-np.log(np.maximum(pr,1e-12))),0)));bs.append(np.sum((pr-s)**2));sa.append(s[r['pred_idx']])
            if r['qtype']=='score':
                order=sorted(range(len(r['labels'])),key=lambda j:int(r['labels'][j]));q=pr[order];target=s[order]
                rps.append(np.mean((np.cumsum(q)[:-1]-np.cumsum(target)[:-1])**2));mae.append(abs(np.dot(np.arange(len(q)),q)-np.dot(np.arange(len(q)),target)))
        result.update(soft_n=len(soft),soft_kl=float(np.mean(kl)),soft_brier=float(np.mean(bs)),soft_accuracy=float(np.mean(sa)))
        if rps:result.update(ordinal_rps_soft=float(np.mean(rps)),ordinal_mae_soft=float(np.mean(mae)))
    hard_score=[r for r in rows if r['qtype']=='score']
    if hard_score:
        rps=[];mae=[]
        for r in hard_score:
            order=sorted(range(len(r['labels'])),key=lambda j:int(r['labels'][j]));pr=r['probs'][order];rank=order.index(r['gold_idx']);onehot=np.eye(len(pr))[rank]
            rps.append(np.mean((np.cumsum(pr)[:-1]-np.cumsum(onehot)[:-1])**2));mae.append(abs(np.dot(np.arange(len(pr)),pr)-rank))
        result.update(ordinal_rps_hard=float(np.mean(rps)),ordinal_mae_hard=float(np.mean(mae)))
    return result

def bootstrap_cases(rows,rng):
    grouped=defaultdict(lambda:defaultdict(list))
    for r in rows: grouped[r['domain']][key(r)].append(r)
    out=[]
    for d in sorted(grouped):
        blocks=list(grouped[d].values())
        for i in rng.integers(0,len(blocks),len(blocks)):out.extend(blocks[i])
    return out

def write_csv(path, records):
    if not records:return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)

def load(path):
    rows=[]; issues=Counter(); seen=set()
    with open(path,newline='') as f:
        for rec in csv.DictReader(f):
            r={k:rec[k] for k in ['model','domain','row_id','question','qtype']};r['labels']=json.loads(rec['labels_json']);r['probs']=np.array(json.loads(rec['probs_json']),float);r['gold_idx']=int(rec['gold_idx']);r['pred_idx']=int(rec['pred_idx'])
            pr=r['probs'];assert len(pr)==len(r['labels']) and len(set(r['labels']))==len(pr)
            assert np.all(np.isfinite(pr)) and np.all(pr>=0) and abs(pr.sum()-1)<1e-4
            assert 0<=r['gold_idx']<len(pr) and 0<=r['pred_idx']<len(pr)
            assert qkey(r) not in seen;seen.add(qkey(r));pr/=pr.sum()
            if r['pred_idx']!=pr.argmax():issues['stored_prediction_tie_other_than_first']+=1;assert abs(pr[r['pred_idx']]-pr.max())<1e-12
            if rec.get('gold_probs_json'):r['gold_probs']=json.loads(rec['gold_probs_json'])
            rows.append(r)
    return rows,dict(issues)

def hydrate(rows):
    gold={}
    for p in (ROOT/'provenance').glob('*_test.json'):
        d=json.loads(p.read_text())
        for rec in d['rows']:
            r=rec['row'];domain='typed_'+r['workflow'];g=json.loads(r['gold'])
            for q,gr in g.items():gold[(domain,str(r['id']),q)]=gr
    count=0
    for r in rows:
        if qkey(r) in gold:
            g=gold[qkey(r)];assert str(g['label'])==r['labels'][r['gold_idx']]
            gp=np.array([g['probabilities'][l] for l in r['labels']]);assert abs(gp.sum()-1)<1e-4;gp/=gp.sum();r['gold_probs']=gp.tolist();count+=1
    return count

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',default=str(ROOT/'data'));ap.add_argument('--out',default=str(ROOT/'results_verified'));ap.add_argument('--seeds',type=int,default=20);ap.add_argument('--bootstrap',type=int,default=1000);ap.add_argument('--models',nargs='+',default=old.DEFAULT_MODELS);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);audits={};all_base=[];experiments=[];loo=[];cis=[];splits={};temps_all={};aligned=None
    for model in args.models:
        path=Path(args.data_dir)/('predictions_'+model+'.csv');rows,issues=load(path);hydrated=hydrate(rows)
        keys={qkey(r) for r in rows}
        if aligned is None:aligned=keys
        else:assert aligned==keys,'Models do not cover identical cases'
        ind=[r for r in rows if r['domain'].startswith('typed_')];ood=[r for r in rows if not r['domain'].startswith('typed_')]
        if ind:
            a,b=old.stratified_split(ind,.2,old.SEED);leaked=len({key(r) for r in a}&{key(r) for r in b})
        else:leaked=0
        audits[model]={'rows':len(rows),'cases':len({key(r) for r in rows}),'old_split_case_overlap':leaked,'argmax_issues':issues,'soft_gold_attached':hydrated,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ood_labels':{d:dict(Counter(r['labels'][r['gold_idx']] for r in ood if r['domain']==d)) for d in sorted({r['domain'] for r in ood})}}
        for name,rr in [('in-domain',ind),('ood',ood)]:
            if rr:
                all_base.append(dict(model=model,domain=name,qtype='all',**stats(rr)))
                for qt in sorted({r['qtype'] for r in rr}):all_base.append(dict(model=model,domain=name,qtype=qt,**stats([r for r in rr if r['qtype']==qt])))
        # Uniform probabilities: no-discrimination reference, canonical option-order ties.
        uni=[dict(r,probs=np.ones(len(r['probs']))/len(r['probs']),pred_idx=0) for r in rows]
        all_base.append(dict(model=model,domain='all',qtype='uniform_reference',**stats(uni)))
        for seed in range(args.seeds):
            if not ind:continue
            a,b=split_cases(ind,.2,20261009+seed)
            if seed==0:splits[model]={'fit_case_ids':[list(k) for k in sorted({key(r) for r in a})],'eval_case_ids':[list(k) for k in sorted({key(r) for r in b})]}
            for mode in ['global','per_type','per_task']:
                t=fit(a,mode)
                for dg,rr in [('in-domain',b),('ood',ood)]:
                    if not rr or (mode=='per_task' and dg=='ood'):continue
                    pp=scale(rr,t,mode);baseline=stats(rr);post=stats(pp)
                    assert abs(post['accuracy']-baseline['accuracy'])<1e-12
                    for qt in ['all']+sorted({r['qtype'] for r in rr}):
                        br=rr if qt=='all' else [r for r in rr if r['qtype']==qt];pr=pp if qt=='all' else [r for r in pp if r['qtype']==qt];bs=stats(br);ps=stats(pr)
                        experiments.append(dict(model=model,seed=seed,eval=dg,mode=mode,qtype=qt,n_fit=len(a),n_eval=len(br),**{'before_'+k:v for k,v in bs.items()},**{'after_'+k:v for k,v in ps.items()},**{'delta_'+k:ps[k]-v for k,v in bs.items() if k in ps and k!='n'}))
                if seed==0:temps_all[model+'/'+mode]={str(k):v for k,v in t.items()}
            print(model,'seed',seed+1,'/',args.seeds,flush=True)
        if ind:
            for hold in sorted({r['domain'] for r in ind}):
                a=[r for r in ind if r['domain']!=hold];b=[r for r in ind if r['domain']==hold]
                for mode in ['global','per_type']:
                    p=scale(b,fit(a,mode),mode)
                    for qt in ['all']+sorted({r['qtype'] for r in b}):
                        bs=stats(b if qt=='all' else [r for r in b if r['qtype']==qt]);ps=stats(p if qt=='all' else [r for r in p if r['qtype']==qt]);loo.append(dict(model=model,held_workflow=hold,mode=mode,qtype=qt,**{'before_'+k:v for k,v in bs.items()},**{'after_'+k:v for k,v in ps.items()}))
            a,b=split_cases(ind,.2,20261009)
            # Two-stage case bootstrap includes uncertainty of the fitted temperature.
            for mode in ['global','per_type']:
                for dg,rr in [('in-domain',b),('ood',ood)]:
                    if not rr:continue
                    bs=stats(rr);ps=stats(scale(rr,fit(a,mode),mode));deltas=defaultdict(list);rng=np.random.default_rng(1337)
                    for i in range(args.bootstrap):
                        aa=bootstrap_cases(a,rng);bb=bootstrap_cases(rr,rng);ss=stats(bb,full=False);tt=stats(scale(bb,fit(aa,mode),mode),full=False)
                        for metric in METRICS:deltas[metric].append(tt[metric]-ss[metric])
                    for metric,v in deltas.items():
                        lo,hi=np.quantile(v,[.025,.975]);cis.append(dict(model=model,mode=mode,eval=dg,metric=metric,before=bs[metric],after=ps[metric],delta=ps[metric]-bs[metric],ci_low=lo,ci_high=hi,B=args.bootstrap,interpretation='paired cluster percentile; refit on resampled disjoint fit cases'))
                    print(model,mode,dg,'bootstrap done',flush=True)
    write_csv(out/'as_shipped_verified.csv',all_base);write_csv(out/'repeated_group_split.csv',experiments);write_csv(out/'leave_one_workflow_out.csv',loo);write_csv(out/'paired_cluster_bootstrap.csv',cis)
    (out/'audit.json').write_text(json.dumps(audits,indent=2));(out/'splits_seed0.json').write_text(json.dumps(splits,indent=2));(out/'temperatures_seed0.json').write_text(json.dumps(temps_all,indent=2));(out/'run_config.json').write_text(json.dumps(dict(vars(args),python=sys.version,numpy=np.__version__,platform=platform.platform(),delta_sign='after minus before; negative is improvement for losses/ECE',note='Repeated split ranges are stability summaries, not confidence intervals. CI refits temperatures. No per-task OOD transfer is reported.'),indent=2))
    print('Saved',out,flush=True)
if __name__=='__main__':main()
