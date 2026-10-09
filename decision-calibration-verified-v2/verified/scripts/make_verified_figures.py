import argparse,csv,sys,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent));import verify_analysis as V

def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder');a=ap.parse_args();folder=Path(a.folder);figdir=folder/'figures';figdir.mkdir(exist_ok=True)
 with open(folder/'paired_cluster_bootstrap.csv') as f:ci=list(csv.DictReader(f))
 data_dir=Path(json.loads((folder/'run_config.json').read_text())['data_dir'])
 models=list(dict.fromkeys(r['model'] for r in ci));fig,axs=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
 for ax,dg in zip(axs,['in-domain','ood']):
  for j,mode in enumerate(['global','per_type']):
   rr=[r for r in ci if r['metric']=='ece_mass_15' and r['eval']==dg and r['mode']==mode];ys=[];means=[];lows=[];highs=[]
   for i,m in enumerate(models):
    r=next(x for x in rr if x['model']==m);ys.append(i+(j-.5)*.18);means.append(float(r['delta']));lows.append(float(r['ci_low']));highs.append(float(r['ci_high']))
   for y,point,lo,hi in zip(ys,means,lows,highs):
    ax.plot([lo,hi],[y,y],color=['#2563eb','#ea580c'][j],lw=2);ax.plot(point,y,'o',color=['#2563eb','#ea580c'][j])
   ax.plot([],[],'o-',color=['#2563eb','#ea580c'][j],label=mode)
  ax.axvline(0,color='#888',ls='--',lw=1);ax.set_yticks(range(len(models)),models);ax.invert_yaxis();ax.set_xlabel('ECE after − before (15 equal-mass bins)');ax.set_title(dg+'; case-disjoint split');ax.grid(axis='x',alpha=.2);ax.legend()
 fig.suptitle('Paired cluster bootstrap: temperatures refitted in every replicate',fontsize=11)
 for ext in ['png','pdf']:fig.savefig(figdir/('calibration_delta_ci.'+ext),dpi=220)
 plt.close(fig)
 fig,axs=plt.subplots(len(models),3,figsize=(10,3*len(models)),layout='constrained',squeeze=False)
 for i,m in enumerate(models):
  rows,_=V.load(data_dir/('predictions_'+m+'.csv'));ind=[r for r in rows if r['domain'].startswith('typed_')];a,b=V.split_cases(ind,.2,20261009);p=V.scale(b,V.fit(a,'per_type'),'per_type')
  for j,qt in enumerate(['noul','choice','score']):
   ax=axs[i,j]
   for rr,label,color in [(b,'shipped','#2563eb'),(p,'per-type refit','#ea580c')]:
    ss=[r for r in rr if r['qtype']==qt];score,bd=V.old.M.ece(ss,15);ax.plot(bd['conf'],bd['acc'],'o-',ms=3,lw=1,color=color,label=f'{label}; ECE={score:.3f}')
   ax.plot([0,1],[0,1],color='#888',ls='--',lw=1);ax.set(xlim=(0,1),ylim=(0,1),title=m+' / '+qt,xlabel='Top-label confidence',ylabel='Reference agreement');ax.legend(fontsize=7);ax.grid(alpha=.15)
 fig.suptitle('Reliability on held-out cases (seed 20261009; equal-mass bins)',fontsize=12)
 for ext in ['png','pdf']:fig.savefig(figdir/('reliability_case_disjoint.'+ext),dpi=180)
 plt.close(fig);print(figdir)
if __name__=='__main__':main()
