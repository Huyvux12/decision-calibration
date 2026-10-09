import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
C0=list(csv.DictReader((ROOT.parent/'checkpoint_omni/taskwise/taskwise_bootstrap.csv').open()));C1=list(csv.DictReader((ROOT/'taskwise_d1-3b/taskwise_bootstrap.csv').open()))
ds=['ood_agnews_balanced','ood_rotten_balanced_v2','ood_sst2_balanced_v2','ood_yelp_rating'];labels=['AG News','Rotten Tomatoes','SST-2','Yelp rating']
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for j,(model,rows) in enumerate([('d1-omni',C0),('d1-3B',C1)]):
 for i,metric in enumerate(['ece_mass_15','nll']):
  ax=axs[i,j]
  for n,(mode,color) in enumerate([('global','#2563eb'),('per_type','#ea580c')]):
   for k,d in enumerate(ds):
    r=next(x for x in rows if x['domain']==d and x['metric']==metric and x['comparison']=='ID_'+mode+'_minus_shipped');y=k+(n-.5)*.18;point=float(r['delta']);lo=float(r['ci_low']);hi=float(r['ci_high'])
    ax.plot([lo,hi],[y,y],color=color,lw=2);ax.plot(point,y,'o',ms=5,color=color)
   ax.plot([],[],'o-',color=color,label=mode.replace('_','-'))
  ax.axvline(0,color='#777',ls='--',lw=1);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_title(model+' — '+('ECE15' if i==0 else 'NLL'));ax.set_xlabel('ID refit − shipped; positive is worse');ax.grid(axis='x',alpha=.2)
  if i==0:ax.legend(fontsize=8,loc='best')
fig.suptitle('Task-specific transfer of workflow-fitted temperatures\n95% paired bootstrap intervals, with temperatures refitted',fontsize=12)
for ext in ['png','pdf']:fig.savefig(ROOT/('taskwise_temperature_effects.'+ext),dpi=220)
plt.close(fig)
print('Saved task-specific figures')
