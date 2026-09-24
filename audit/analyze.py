"""Audit checks C1-C4 on the inherited pools: trace fidelity, feature
diagnostics, constant-value interventions, exact paired tests, monotonicity
grid and observed-input counterfactuals. Leaves application results unchanged."""
from pathlib import Path
import sys, json, math, hashlib, itertools, os
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
sys.path.insert(0,str(REPO/'src'))
from ogd_ir.model import Corpus
from ogd_ir.ranking import Ranker
from ogd_ir.evaluation import metrics
from ogd_ir.io import digest

def save(name,data):
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results'/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def array_infer(v,cfg):
    """Independent array evaluation of the declared default rule policy."""
    def tri(x,points):
        a,b,c=points
        ascending=np.ones_like(x) if a==b else (x-a)/(b-a)
        descending=np.ones_like(x) if b==c else (c-x)/(c-b)
        return np.maximum(0,np.minimum(1,np.minimum(ascending,descending)))
    feats=['similarity','completeness','freshness','resources']; terms=['low','medium','high']
    memberships=np.stack([np.stack([tri(v[:,i],cfg['memberships'][f][t]) for t in terms],axis=1) for i,f in enumerate(feats)],axis=1)
    strengths=np.zeros((len(v),5))
    for n,inds in enumerate(itertools.product(range(3),repeat=4)):
        s,c,t,a=inds;out=0 if s==0 else s+min(c,max(t,a))
        strength=np.min(np.stack([memberships[:,i,k] for i,k in enumerate(inds)],axis=1),axis=1)
        strength=np.minimum(1,strength*cfg['rule_weights'].get(f'R{n:03}',1))
        strengths[:,out]=np.maximum(strengths[:,out],strength)
    u=np.linspace(0,1,cfg['universe_points']); curves=np.stack([tri(u,cfg['output_memberships'][t]) for t in ['very_low','low','moderate','good','excellent']])
    scores=[]
    for start in range(0,len(v),1024):
        ss=strengths[start:start+1024];agg=np.zeros((len(ss),len(u)))
        for i in range(5):agg=np.maximum(agg,np.minimum(ss[:,i,None],curves[i]))
        assert np.all(agg.sum(1)>0)
        scores.extend(((agg*u).sum(1)/agg.sum(1)).tolist())
    return np.array(scores)

def stats(v):
    return {k:float(x) for k,x in dict(min=np.min(v),median=np.median(v),max=np.max(v),mean=np.mean(v),std=np.std(v)).items()}|{'unique':len(set(v))}

def main():
    c=Corpus.load(REPO/'data/legacy/corpus_v2.json');r=Ranker(c)
    old=json.loads((REPO/'results/application_reanalysis.json').read_text())
    qrels=json.loads((REPO/'data/legacy/judgments.json').read_text()); qs=qrels['queries'];feats=['similarity','completeness','freshness','resources']
    assert c.hash==old['corpus_hash'] and r.policy_hash==old['config_hash']
    if digest(qrels)!=old['judgments_hash']:
        # An anonymised review copy replaces names in the provenance block, which changes this digest.
        if os.environ.get('OGD_ALLOW_PROVENANCE_EDIT')!='1':
            raise SystemExit('Judgment digest differs from the application report; set OGD_ALLOW_PROVENANCE_EDIT=1 for an anonymised copy')
        grades=hashlib.sha256(json.dumps([[q['id'],sorted(q['judgments'].items())] for q in qs]).encode()).hexdigest()
        assert grades=='7278fcffa2236cb85f2af339f4272dc0cf83004e572e6c8bf00647fbb098bc27', 'Inherited grades differ from the evaluated ones'
    diag=[]
    for d in c.datasets:
        f,e=r.features('',d);diag.append({'dataset_id':d.id,'publisher':d.publisher,**f,**e})
    metadata={'datasets':len(diag),'config_hash':r.policy_hash,'corpus_hash':c.hash,'reference_date':r.config['reference_date'],
      'features':{f:stats([x[f] for x in diag]) for f in feats[1:]},
      'modified_dates':dict(Counter(str(x['metadata_modified'])[:10] for x in diag)),
      'completeness_counts':dict(Counter(x['completeness'] for x in diag)),
      'checks_present':{k:sum(x['completeness_checks'][k] for x in diag) for k in diag[0]['completeness_checks']},
      'age_days':stats([x['age_days'] for x in diag]),'resource_counts':dict(Counter(x['resource_count'] for x in diag)),
      'publisher_features':{p:{f:stats([x[f] for x in diag if x['publisher']==p]) for f in feats[1:]} for p in sorted({x['publisher'] for x in diag})},
      'rows':diag}
    save('feature_diagnostics.json',metadata);print('Feature diagnostics complete',flush=True)
    traces=[];errors=[];pool_scores={};topruns={};coverage=[]
    for q in qs:
        scored=[]
        for d in c.datasets:
            trace=r.score(q['text'],d.id);scored.append(trace)
            if d.id in q['judgments']:traces.append({'query_id':q['id'],**trace})
        vals=np.array([[t['features'][f] for f in feats] for t in scored]); independent=array_infer(vals,r.config)
        errors.extend(abs(independent-np.array([t['inference']['centroid'] for t in scored])))
        for t in scored:
            assert abs(sum(x['bm25_contribution'] for x in t['evidence']['matched_terms'])-t['evidence']['bm25'])<1e-10
            linear=sum(t['linear_contributions'].values()); hybrid=r.config['alpha']*t['inference']['centroid']+(1-r.config['alpha'])*linear
            assert abs(hybrid-t['ungated_hybrid'])<1e-12
            assert abs(t['scores']['hybrid']-(hybrid if t['lexical_gate_passed'] else 0))<1e-12
        pool_scores[q['id']]={t['dataset_id']:t for t in scored if t['dataset_id'] in q['judgments']}
        topruns[q['id']]={}
        for s in ['bm25','linear','mamdani','hybrid']:
            pool=sorted(pool_scores[q['id']],key=lambda d:(-pool_scores[q['id']][d]['scores'][s],d))
            assert pool==old['runs'][q['id']][s]
            order=[t['dataset_id'] for t in sorted(scored,key=lambda t:(-t['scores'][s],t['dataset_id'])) if t['lexical_gate_passed']][:10]
            topruns[q['id']][s]=order
            coverage.append({'query_id':q['id'],'system':s,'returned':len(order),'judged':sum(d in q['judgments'] for d in order),'unjudged':sum(d not in q['judgments'] for d in order)})
    save('pooled_traces.json',traces)
    save('trace_validation.json',{'query_document_pairs':len(errors),'max_absolute_centroid_error':float(max(errors)),'verified_pooled_orders':60,'pooled_traces':len(traces),'checks':['Independent array centroid versus scorer','BM25 matched-term contribution sums','Hybrid arithmetic and lexical gate','Saved pooled order equality'], 'not_tested':'Human comprehension or normative appropriateness of the policy'})
    print('7500 traces checked; lexical candidate coverage collected',flush=True)
    # Intervention 1 fixes a feature to its corpus median; intervention 2 fixes it to 0.5.
    ablations=[]
    for target in ['corpus_median','0.5']:
        for f in feats[1:]:
            fixed=float(np.median([d[f] for d in diag])) if target=='corpus_median' else .5
            rows=[]
            for q in qs:
                current=pool_scores[q['id']];ids=list(current)
                v=np.array([[fixed if x==f else current[d]['features'][x] for x in feats] for d in ids])
                fuzzy=array_infer(v,r.config);linear=v.mean(1);hybrid=r.config['alpha']*fuzzy+(1-r.config['alpha'])*linear
                scores={d:float(hybrid[i]) if current[d]['lexical_gate_passed'] else 0 for i,d in enumerate(ids)}
                order=sorted(ids,key=lambda d:(-scores[d],d));m=metrics(order,q['judgments'])
                rows.append({'query_id':q['id'],**m,'changed_order':order!=old['runs'][q['id']]['hybrid'],'order':order})
            ablations.append({'intervention':target,'feature':f,'fixed_value':fixed,'mean_ndcg10':float(np.mean([x['ndcg10'] for x in rows])),
               'delta_from_default':float(np.mean([x['ndcg10'] for x in rows]))-old['aggregates']['all_queries']['hybrid']['ndcg10']['mean'],
               'changed_queries':sum(x['changed_order'] for x in rows),'per_query':rows})
    save('feature_interventions.json',{'scope':'Ten fixed inherited candidates, same gate; constant-value interventions, not feature removal or tuning','baseline_ndcg10':old['aggregates']['all_queries']['hybrid']['ndcg10']['mean'],'variants':ablations})
    # Exact sign-flip paired mean test; each independent query is the exchangeability unit.
    exact=[]
    for a,b in itertools.combinations(old['systems'],2):
        va={x['query_id']:x['ndcg10'] for x in old['per_query'] if x['system']==a};vb={x['query_id']:x['ndcg10'] for x in old['per_query'] if x['system']==b}
        delta=np.array([va[q['id']]-vb[q['id']] for q in qs]);delta[np.abs(delta)<1e-12]=0; nonzero=delta[delta!=0]
        signs=np.array(list(itertools.product([-1,1],repeat=len(nonzero))))
        null=signs@nonzero/len(delta);observed=float(delta.mean());p=float(np.mean(np.abs(null)>=abs(observed)-1e-12))
        exact.append({'a':a,'b':b,'mean_difference':observed,'nonzero_query_pairs':len(nonzero),'distinct_sign_assignments':2**len(nonzero),'p_exact':p})
    running=0
    for i,t in enumerate(sorted(exact,key=lambda x:x['p_exact'])):
        running=max(running,min(1,(len(exact)-i)*t['p_exact']));t['p_holm']=running
    save('exact_paired_tests.json',{'statistic':'Absolute paired mean nDCG@10 difference','two_sided':'Exhaustive within-query system-label swaps; zero pairs factored out; tolerance 1e-12','family':'All ten pairs of five systems, 15 queries','assumption':'Independent query-level exchangeability under the null. A small selected query set limits generalization.','tests':exact})
    print('Interventions and exact paired tests complete',flush=True)
    # Ordered linguistic consequents do not by themselves establish score monotonicity.
    levels=np.linspace(0,1,21);grid=np.array(list(itertools.product(levels,repeat=4)))
    F=array_infer(grid,r.config); H=.65*F+.35*grid.mean(1); monotone=[]
    for system,out in [('mamdani',F),('hybrid',H)]:
        cubes=out.reshape((21,)*4)
        for axis,f in enumerate(feats):
            dif=np.diff(cubes,axis=axis);neg=dif < -1e-10;ix=np.unravel_index(np.argmin(dif),dif.shape);hi=list(ix);hi[axis]+=1
            lowv={ff:float(levels[k]) for ff,k in zip(feats,ix)};highv={ff:float(levels[k]) for ff,k in zip(feats,hi)}
            monotone.append({'system':system,'feature':f,'adjacent_pairs':dif.size,'decreases':int(neg.sum()),'largest_decrease':float(min(0,dif.min())),
               'example_low':lowv,'example_high':highv,'example_score_low':float(cubes[ix]),'example_score_high':float(cubes[tuple(hi)])})
    save('monotonicity_grid.json',{'scope':'Synthetic feature grid, not observed user behaviour or effectiveness','grid_points':len(grid),'step':.05,'score':'Ungated outputs. A synthetic positive lexical gate is assumed for hybrid.','tests':monotone})
    # Counterfactual increases at observed candidate features, including passing gates only.
    observed=[]
    passing=[t for t in traces if t['lexical_gate_passed']]
    v=np.array([[t['features'][f] for f in feats] for t in passing]);F0=array_infer(v,r.config);H0=.65*F0+.35*v.mean(1)
    for axis,f in enumerate(feats):
        newer=v.copy();newer[:,axis]=np.minimum(1,newer[:,axis]+.05);F1=array_infer(newer,r.config);H1=.65*F1+.35*newer.mean(1)
        for system,diffs in [('mamdani',F1-F0),('hybrid',H1-H0)]:
            i=int(np.argmin(diffs));observed.append({'feature':f,'system':system,'tested_pairs':len(passing),'decreases':int((diffs < -1e-10).sum()),'largest_decrease':float(min(0,diffs.min())),
              'example_query_id':passing[i]['query_id'],'example_dataset_id':passing[i]['dataset_id'],'example_before':v[i].tolist(),'example_after':newer[i].tolist()})
    save('observed_feature_counterfactuals.json',{'scope':'Hypothetical +0.05 feature intervention on positively gated historical pairs. Does not alter stored metadata.','tests':observed})
    save('full_corpus_candidate_coverage.json',{'systems':['bm25','linear','mamdani','hybrid'],'top10_runs':topruns,'coverage':coverage,'scope':'Coverage only; unjudged results have no relevance grade.'})
    save('audit_manifest.json',{'config_hash':r.policy_hash,'corpus_hash':c.hash,'judgments_hash':digest(qrels),'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (REPO/'src/ogd_ir').glob('*.py')},'source_report_sha256':hashlib.sha256((REPO/'results/application_reanalysis.json').read_bytes()).hexdigest()})
    print('Synthetic monotonicity and observed counterfactuals complete',flush=True)

if __name__=='__main__':main()
