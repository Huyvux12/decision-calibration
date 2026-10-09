"""Colab GPU extension: balanced OOD incl. ordinal ratings, resume, pinned provenance."""
import argparse, csv, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time
from pathlib import Path
os.environ.setdefault('USE_TF','0');os.environ.setdefault('TRANSFORMERS_NO_TF','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from decision_calib import models, datasets as D
FIELDS=['model','domain','row_id','question','qtype','gold_idx','pred_idx','labels_json','probs_json','gold_probs_json']

def balanced(ds,quota,seed):
    labels=sorted(set(ds['label']));rng=np.random.default_rng(seed);selected=[]
    for label in labels:
        ids=np.flatnonzero(np.array(ds['label'])==label)
        assert len(ids)>=quota
        selected.extend(rng.choice(ids,quota,replace=False).tolist())
    # Gold is used only for stratified data selection and evaluation, never model input.
    return [(i,ds[i]) for i in sorted(selected)]

def get_cases(quota,yelp_quota,typed,pins):
    from datasets import load_dataset
    cases=[];metadata={}
    def get(repo,config,split):
        ds=load_dataset(repo,config,split=split,revision=pins['datasets'][repo]);metadata[repo+'/'+str(config)+'/'+split]={'revision':pins['datasets'][repo],'fingerprint':ds._fingerprint,'n_source':len(ds)};return ds
    def add(domain,i,state,qs,gs):
        cases.append(dict(domain=domain,row_id=str(i),state=state,questions=qs,gold=gs))
    if typed:
        for wf in D.TYPED_WORKFLOWS:
            ds=get('LocalLLaMA/typed-decisions',wf,'test')
            for r in ds:add('typed_'+wf,r['id'],json.dumps(json.loads(r['state'])),json.loads(r['questions']),json.loads(r['gold']))
    for repo,cfg,split,domain,field,qs in [
        ('fancyzhx/ag_news',None,'test','ood_agnews_balanced','text',D.AGNEWS_Q),
        ('cornell-movie-review-data/rotten_tomatoes',None,'test','ood_rotten_balanced_v2','text',D.SENTIMENT_Q),
        ('nyu-mll/glue','sst2','validation','ood_sst2_balanced_v2','sentence',D.SENTIMENT_Q)]:
        ds=get(repo,cfg,split)
        for i,r in balanced(ds,quota,20261009):
            if 'agnews' in domain:gold={'topic':{'type':'choice','label':D.AGNEWS_LABELS[int(r['label'])]}}
            else:gold={'positive':{'type':'noul','label':'true' if int(r['label'])==1 else 'false'}}
            add(domain,i,r[field],qs,gold)
    # A new semantic task, not a noisy relabeling of movie sentiment.
    qs={'rating':{'type':'score','instructions':'What star rating did the author give this business, based on the review?', 'criteria':['1 star: very dissatisfied','2 stars: dissatisfied','3 stars: mixed or neutral','4 stars: satisfied','5 stars: very satisfied']}}
    ds=get('Yelp/yelp_review_full',None,'test')
    for i,r in balanced(ds,yelp_quota,20261009):
        assert 0<=int(r['label'])<5
        # Decision Index levels are 0..4; rank 0 corresponds to one star.
        add('ood_yelp_rating',i,r['text'],qs,{'rating':{'type':'score','label':str(int(r['label']))}})
    return cases,metadata

def freeze_pins(out):
    from huggingface_hub import HfApi
    path=out/'resolved_revisions.json'
    if path.exists():return json.loads(path.read_text())
    api=HfApi();pins={'models':{},'datasets':{},'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    for m in ['d1-omni','d1-3b','opendecider-small','jeff-2b','jeff-0.8b']:
        repo=models.REGISTRY[m]['repo'];pins['models'][repo]=api.model_info(repo).sha
    pins['models']['Qwen/Qwen3-4B-Instruct-2507']=api.model_info('Qwen/Qwen3-4B-Instruct-2507').sha
    for repo in ['LocalLLaMA/typed-decisions','fancyzhx/ag_news','cornell-movie-review-data/rotten_tomatoes','nyu-mll/glue','Yelp/yelp_review_full']:
        pins['datasets'][repo]=api.dataset_info(repo).sha
    # Pin Jeff's external adapter code as well as its weights.
    clone=out/'jeff_src';subprocess.run(['git','clone','--depth','1','https://github.com/firelex/jeff',str(clone)],check=True)
    pins['jeff_commit']=subprocess.check_output(['git','-C',str(clone),'rev-parse','HEAD'],text=True).strip();pins['jeff_source']=str(clone.resolve())
    path.write_text(json.dumps(pins,indent=2));return pins

def load_pinned(name,pins,out):
    import torch
    from transformers import AutoModel
    from huggingface_hub import snapshot_download
    entry=models.REGISTRY[name];repo=entry['repo']
    if name.startswith('d1'):
        checkpoint=snapshot_download(repo,revision=pins['models'][repo])
        model=AutoModel.from_pretrained(checkpoint,trust_remote_code=True,dtype=torch.float16).to('cuda').eval()
        return entry,model
    if name.startswith('jeff'):
        src=Path(pins['jeff_source'])
        if not src.exists():
            subprocess.run(['git','clone','https://github.com/firelex/jeff',str(src)],check=True);subprocess.run(['git','-C',str(src),'checkout',pins['jeff_commit']],check=True)
        models._JEFF_SRC=str(src/'src');sys.path.insert(0,models._JEFF_SRC)
        checkpoint=snapshot_download(repo,revision=pins['models'][repo]);from jeff.model import DecisionModel
        return entry,DecisionModel(checkpoint=checkpoint)
    if name=='opendecider-small':
        # Restrict every Hub call inside original adapter to resolved revisions.
        import huggingface_hub,transformers
        original=huggingface_hub.hf_hub_download
        def frozen_download(repo_id,*a,**kw):kw['revision']=pins['models'][repo_id];return original(repo_id,*a,**kw)
        huggingface_hub.hf_hub_download=frozen_download
        patches=[]
        for cls in [transformers.AutoTokenizer,transformers.AutoModelForCausalLM]:
            orig=cls.from_pretrained
            def frozen(repo_id,*a,_orig=orig,**kw):kw['revision']=pins['models'][repo_id];return _orig(repo_id,*a,**kw)
            patches.append((cls,orig));cls.from_pretrained=frozen
        try:handle=entry['load']()
        finally:
            huggingface_hub.hf_hub_download=original
            for cls,orig in patches:cls.from_pretrained=orig
        return entry,handle
    raise ValueError(name)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['d1-omni','d1-3b','opendecider-small','jeff-2b','jeff-0.8b'],default='d1-omni');ap.add_argument('--out',default=str(ROOT/'extension_output'));ap.add_argument('--quota',type=int,default=250);ap.add_argument('--yelp-per-class',type=int,default=200);ap.add_argument('--include-typed',action='store_true');ap.add_argument('--probe',action='store_true');args=ap.parse_args()
    assert 1<=args.quota<=400 and 1<=args.yelp_per_class<=10000
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);pins=freeze_pins(out)
    cases,meta=get_cases(args.quota,args.yelp_per_class,args.include_typed,pins)
    if args.probe:
        # Probe every domain and every typed primitive, not only the head of one loader.
        grouped={}
        for c in cases:grouped.setdefault(c['domain'],c)
        cases=list(grouped.values())
    snapshot=out/'cases.jsonl';content=''.join(json.dumps(c,ensure_ascii=False,sort_keys=True)+'\n' for c in cases)
    casehash=hashlib.sha256(content.encode()).hexdigest();snapshot=out/('cases_probe.jsonl' if args.probe else 'cases.jsonl')
    if snapshot.exists():assert snapshot.read_text()==content,'Dataset changed; use a new output folder'
    else:snapshot.write_text(content)
    fname=out/('probe_'+args.model+'.csv' if args.probe else 'predictions_'+args.model+'.csv');done=set(); existing=0
    if fname.exists():
        with open(fname,newline='') as f:
            for r in csv.DictReader(f):
                k=(r['domain'],r['row_id'],r['question']);assert k not in done;done.add(k);existing+=1
    config_path=out/('probe_' if args.probe else '')/('config_'+args.model+'.json');config_path.parent.mkdir(exist_ok=True)
    config=dict(vars(args),cases_sha256=casehash)
    if config_path.exists():assert json.loads(config_path.read_text())==config,'Resume configuration differs'
    else:config_path.write_text(json.dumps(config,indent=2))
    expected={(c['domain'],c['row_id'],qn) for c in cases for qn in c['gold']};assert done<=expected
    entry,handle=load_pinned(args.model,pins,out);import torch
    t0=time.time();errors=[];n=existing
    with open(fname,'a' if fname.exists() else 'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if not existing:w.writeheader()
        for ci,c in enumerate(cases):
            if all((c['domain'],c['row_id'],q) in done for q in c['gold']):continue
            try:
                with torch.inference_mode():raw=entry['predict_batch'](handle,[(c['state'],c['questions'])],chunk=1)
                assert len(raw)==1
                for qn,g in c['gold'].items():
                    k=(c['domain'],c['row_id'],qn)
                    if k in done:continue
                    norm=entry['normalize'](raw[0][qn],g['type']);labels=sorted(norm['probs']);pr=np.array([norm['probs'][l] for l in labels],dtype=np.float64)
                    assert np.all(np.isfinite(pr)) and np.all(pr>=0) and abs(pr.sum()-1)<1e-3;pr/=pr.sum()
                    assert str(g['label']) in labels;pred=labels.index(norm['pred']);assert abs(pr[pred]-pr.max())<1e-7
                    gp=g.get('probabilities');gold_probs=[gp[l] for l in labels] if gp else None
                    w.writerow(dict(model=args.model,domain=c['domain'],row_id=c['row_id'],question=qn,qtype=g['type'],gold_idx=labels.index(str(g['label'])),pred_idx=pred,labels_json=json.dumps(labels),probs_json=json.dumps(pr.tolist()),gold_probs_json=json.dumps(gold_probs) if gp else ''));done.add(k);n+=1
                f.flush();os.fsync(f.fileno())
            except Exception as e:
                import traceback
                errors.append(dict(domain=c['domain'],row_id=c['row_id'],error=repr(e),traceback=traceback.format_exc()));print('ERROR',errors[-1],flush=True)
            if ci%25==0:print(args.model,ci+1,'/',len(cases),'rows',n,'seconds',round(time.time()-t0),flush=True)
    versions={}
    for package in ['torch','transformers','numpy','scipy','datasets','huggingface_hub','opendecider','safetensors']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
    manifest=dict(model=args.model,repo=entry['repo'],revision=pins['models'][entry['repo']],cases_sha256=casehash,expected_rows=len(expected),actual_rows=n,missing=[list(k) for k in sorted(expected-done)],errors=errors,versions=versions,python=sys.version,gpu=torch.cuda.get_device_name(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.time()-t0,datasets=meta,complete=done==expected)
    (out/('probe_manifest_' if args.probe else 'manifest_')/(args.model+'.json')).parent.mkdir(exist_ok=True)
    mp=out/('probe_manifest_'+args.model+'.json' if args.probe else 'manifest_'+args.model+'.json');mp.write_text(json.dumps(manifest,indent=2))
    manifest['explicit_config_temperatures']=getattr(handle.config,'temperatures',None) if hasattr(handle,'config') else None
    if args.model=='d1-omni' and manifest['complete']:
        raw_path=out/('raw_probe_'+args.model+'.csv' if args.probe else 'raw_predictions_'+args.model+'.csv')
        with open(fname,newline='') as f:records=list(csv.DictReader(f))
        temps=manifest['explicit_config_temperatures'];assert temps is not None
        for r in records:
            pr=np.array(json.loads(r['probs_json']));k=len(pr);group=r['qtype']+':'+('2' if k<=2 else '3-5' if k<=5 else '6-10' if k<=10 else '11+')
            t=temps.get(group,temps.get(r['qtype'],1.));lp=np.log(np.maximum(pr,1e-12))*t;pr=np.exp(lp-lp.max());pr/=pr.sum();r['probs_json']=json.dumps(pr.tolist())
        with open(raw_path,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(records)
        manifest['raw_arm']=dict(path=str(raw_path),method='Inverse of explicit shipped temperature; text only; limited by float precision')
    mp.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ['model','actual_rows','expected_rows','complete']},indent=2),flush=True)
    if not manifest['complete']:raise RuntimeError('Incomplete run; retain outputs and rerun to resume missing cases')
if __name__=='__main__':main()
