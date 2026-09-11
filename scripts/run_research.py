"""Rebuild every reported result from the same fixed policy and frozen inputs."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import platform
import numpy as np
import scipy
from scipy.stats import kendalltau
from ogd_ir.model import Corpus
from ogd_ir.config import load_config,FEATURES
from ogd_ir.ranking import Ranker,explain
from ogd_ir.evaluation import evaluate
from ogd_ir.io import read_json,write_json,digest
ROOT=Path(__file__).resolve().parents[1]

def main():
 c=Corpus.load(ROOT/'data/legacy/corpus.json'); q=read_json(ROOT/'data/legacy/judgments.json');cfg=load_config(ROOT/'configs/default.json')
 baseline=evaluate(Ranker(c,cfg),q,allow_unverified=True);write_json(ROOT/'results/legacy_reanalysis.json',baseline)
 audit={'datasets':len(c.datasets),'publishers':dict(Counter(d.publisher for d in c.datasets)), 'modification_days':dict(Counter((d.modified or 'missing')[:10] for d in c.datasets)), 'grade_counts':dict(Counter(str(g) for row in q['queries'] for g in row['judgments'].values())), 'zero_positive_queries':[row['id'] for row in q['queries'] if not any(row['judgments'].values())], 'warning':'Zero positives apply to the inherited judged pools, not the full corpus.'}
 write_json(ROOT/'results/corpus_audit.json',audit)
 exposure={}
 import math
 for system in ('bm25','linear','mamdani','hybrid'):
  counts=Counter()
  for runs in baseline['runs'].values():
   for i,identity in enumerate(runs[system]):counts[c.by_id[identity].publisher]+=1/math.log2(i+2)
  total=sum(counts.values());exposure[system]={p:score/total for p,score in sorted(counts.items())}
 write_json(ROOT/'results/publisher_exposure.json',{'scope':'Discounted rank exposure within inherited pools, all queries including zero-positive pools. Not a fairness estimate or corpus-wide visibility measure.','exposure':exposure})
 variants=[]
 for feature in FEATURES:
  for delta in [-.05,.05]:
   v=deepcopy(cfg);v['memberships'][feature]['medium'][1]+=(delta);variants.append((f'{feature}_middle_peak_{delta:+.2f}',v))
 for value in [.5,.8]:
  v=deepcopy(cfg);v['alpha']=value;variants.append((f'alpha_{value}',v))
 for term in ['low','moderate','good','excellent']:
  v=deepcopy(cfg)
  from ogd_ir.ranking import rules
  v['rule_weights']={r['id']:.9 for r in rules(v) if r['then']==term};variants.append((f'{term}_rule_weights_0.9',v))
 sensitivity=[]
 for name,config in variants:
  result=evaluate(Ranker(c,config),q,allow_unverified=True)
  if result['config_hash']==baseline['config_hash']:raise ValueError('No-op perturbation')
  taus=[];changed=0
  for qid,runs in baseline['runs'].items():
   a=runs['hybrid'];b=result['runs'][qid]['hybrid'];taus.append(float(kendalltau(list(range(len(a))),[b.index(x) for x in a]).statistic));changed+=a!=b
  sensitivity.append({'name':name,'config_hash':result['config_hash'],'config':config,'changed_query_orders':changed,'mean_tau':float(np.mean(taus)),'hybrid_metrics':result['aggregates']['all_queries']['hybrid']})
 write_json(ROOT/'results/sensitivity.json',{'baseline_config_hash':baseline['config_hash'],'baseline_metrics':baseline['aggregates']['all_queries']['hybrid'],'variations':sensitivity,'scope':'14 prespecified local policy perturbations. Kendall tau uses deterministic total ranks including ID tie-breaks. Changed configuration does not guarantee changed ranking.'})
 # Verified contrasting trace on an inherited query: choose largest BM25/hybrid rank movement.
 best=None;r=Ranker(c,cfg)
 for row in q['queries']:
  runs=baseline['runs'][row['id']]
  for identity in runs['bm25']:
   move=runs['bm25'].index(identity)-runs['hybrid'].index(identity)
   if best is None or abs(move)>abs(best[0]):best=(move,row,identity)
 move,row,identity=best
 other=next(x for x in baseline['runs'][row['id']]['hybrid'] if x!=identity)
 traces=[r.score(row['text'],x) for x in dict.fromkeys([identity,other])]
 write_json(ROOT/'results/explanation_example.json',{'query_id':row['id'],'query':row['text'],'rank_movement':move,'traces':traces})
 (ROOT/'results/explanation_example.txt').write_text(row['text']+'\n\n'+'\n\n'.join(explain(t) for t in traces),encoding='utf-8')
 write_json(ROOT/'results/environment.json',{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'source_hash':digest({str(p.relative_to(ROOT)):p.read_text() for p in sorted((ROOT/'src').rglob('*.py'))})})
 print('Aggregate means (all 15 queries):')
 for system,vals in baseline['aggregates']['all_queries'].items():print(system,{k:round(v['mean'],6) for k,v in vals.items() if k!='n'})
 print('Sensitivity tau range',min(v['mean_tau'] for v in sensitivity),max(v['mean_tau'] for v in sensitivity))
if __name__=='__main__':main()
