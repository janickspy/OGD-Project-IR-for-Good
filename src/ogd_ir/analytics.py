"""Metadata diagnostics and within-candidate ranking consequences."""
from collections import Counter
import math
from .config import FEATURES


def corpus_summary(ranker):
    rows=[]
    for d in ranker.corpus.datasets:
        features,evidence=ranker.features('',d)
        rows.append({'dataset_id':d.id,'publisher':d.publisher,**features,
                     'date_status':evidence['date_status'],'age_days':evidence['age_days'],
                     'resource_count':d.resource_count,'modified':d.modified,
                     **{'missing_'+k:not v for k,v in evidence['completeness_checks'].items()}})
    return {'corpus_hash':ranker.corpus.hash,'config_hash':ranker.policy_hash,'reference_date':ranker.config['reference_date'],
            'datasets':len(rows),'publishers':dict(Counter(d.publisher for d in ranker.corpus.datasets)),
            'date_status':dict(Counter(r['date_status'] for r in rows)),
            'missing_fields':{k:sum(r['missing_'+k] for r in rows) for k in ranker.features('',ranker.corpus.datasets[0])[1]['completeness_checks']},
            'rows':rows,'caveat':'Metadata presence, modification age and resource counts are proxies, not intrinsic quality or verified availability.'}


def ranking_changes(ranker, query, candidates=None, limit=10):
    traces=ranker.rank(query,candidates=candidates,limit=len(ranker.corpus.datasets))
    orders={system:sorted(traces,key=lambda t:(-t['scores'][system],t['dataset_id']))
            for system in ('bm25','linear','mamdani','hybrid')}
    baseline={t['dataset_id']:i+1 for i,t in enumerate(orders['bm25'])}
    exposure={};movements=[]
    for system,order in orders.items():
        totals=Counter()
        for rank,t in enumerate(order,1):
            d=ranker.corpus.by_id[t['dataset_id']]
            if rank<=limit:totals[d.publisher]+=1/math.log2(rank+1)
            movements.append({'system':system,'dataset_id':d.id,'title':d.title,'publisher':d.publisher,
                              'rank':rank,'bm25_rank':baseline[d.id],'positions_gained':baseline[d.id]-rank,
                              **t['features']})
        total=sum(totals.values());exposure[system]={p:s/total for p,s in totals.items()}
    return {'query':query,'limit':limit,'exposure':exposure,'movements':movements,
            'corpus_hash':ranker.corpus.hash,'config_hash':ranker.policy_hash,
            'scope':'One query, same selected candidates; discounted rank exposure is descriptive, not a fairness judgment.'}


def policy_ablation(ranker, query, candidates=None, limit=10):
    """Freeze the candidate set, neutralize one feature at a time, re-run inference."""
    traces=ranker.rank(query,candidates=candidates,limit=len(ranker.corpus.datasets))
    result={}
    for feature in FEATURES[1:]:
        scores={}
        for trace in traces:
            values={**trace['features'],feature:.5};fuzzy,_=ranker.infer(values)
            weights=ranker.config['feature_weights'];linear=sum(values[k]*weights[k] for k in FEATURES)/sum(weights.values())
            scores[trace['dataset_id']]=ranker.config['alpha']*fuzzy+(1-ranker.config['alpha'])*linear
        result[feature]=sorted(scores,key=lambda i:(-scores[i],i))[:limit]
    return {'intervention':'Set one normalized metadata feature to 0.5; preserve lexical gate, candidates and remaining policy',
            'baseline':[t['dataset_id'] for t in traces[:limit]],'orders':result,'config_hash':ranker.policy_hash}
