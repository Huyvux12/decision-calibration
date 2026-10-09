"""Exercise inference bookkeeping/resume with a fake adapter; no GPU/model accuracy claim."""
import contextlib,csv,json,sys,tempfile,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import run_extension as E

def main():
 cases=[dict(domain='typed_mock',row_id='case_1',state='public fake test state',questions={'x':{}},gold={'x':{'type':'noul','label':'true'}}),dict(domain='ood_yelp_rating',row_id='case_2',state='public fake test review',questions={'x':{}},gold={'x':{'type':'score','label':'1'}})]
 pins={'models':{'fake/repo':'fake_revision'}}
 E.freeze_pins=lambda out:pins
 E.get_cases=lambda quota,yelp_quota,typed,pins:(cases,{'fake':'test'})
 count={'predict':0}
 def predict(handle,items,chunk):
  count['predict']+=1
  qt='score' if 'review' in items[0][0] else 'noul'
  p={'0':.1,'1':.8,'2':.1} if qt=='score' else {'false':.1,'true':.9}
  return [{'x':{'pred':'1' if qt=='score' else 'true','probs':p}}]
 entry={'repo':'fake/repo','predict_batch':predict,'normalize':lambda answer,qt:answer}
 handle=types.SimpleNamespace(config=types.SimpleNamespace(temperatures={'noul:2':2.,'score:3-5':2.}))
 E.load_pinned=lambda name,pins,out:(entry,handle)
 fake_torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext,cuda=types.SimpleNamespace(get_device_name=lambda:'FAKE bookkeeping test',max_memory_allocated=lambda:0))
 oldtorch=sys.modules.get('torch');sys.modules['torch']=fake_torch;oldargs=sys.argv
 try:
  with tempfile.TemporaryDirectory() as tmp:
   sys.argv=['run_extension.py','--model','d1-omni','--out',tmp,'--quota','1','--yelp-per-class','1','--include-typed'];E.main()
   m=json.loads((Path(tmp)/'manifest_d1-omni.json').read_text());assert m['complete'] and m['actual_rows']==2
   rows=list(csv.DictReader((Path(tmp)/'predictions_d1-omni.csv').open()));assert len(rows)==2
   assert (Path(tmp)/'raw_predictions_d1-omni.csv').is_file()
   assert count['predict']==2;E.main();assert count['predict']==2
   assert len(list(csv.DictReader((Path(tmp)/'predictions_d1-omni.csv').open())))==2
   # Resume must not silently accept changed quotas/samples.
   sys.argv[sys.argv.index('--quota')+1]='2'
   try:E.main()
   except AssertionError:pass
   else:raise AssertionError('Changed configuration was accepted')
 finally:
  sys.argv=oldargs
  if oldtorch is None:sys.modules.pop('torch',None)
  else:sys.modules['torch']=oldtorch
 print('PASS: complete count, manifest, inverse-temperature CSV, resume without duplicate inference, changed-config rejection. Fake adapter only.')
if __name__=='__main__':main()
