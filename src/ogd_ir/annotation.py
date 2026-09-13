"""Create blind, deterministic annotation pools; never auto-fill relevance."""
import random
from .evaluation import SYSTEMS
from .io import write_json

def build_pool(ranker, queries, output=None, depth=20, seed=2026, semantic=None):
    if depth<1:raise ValueError('Depth must be positive')
    if not queries or len({q['id'] for q in queries})!=len(queries):raise ValueError('Invalid query IDs')
    records=[];rng=random.Random(seed)
    for q in sorted(queries,key=lambda q:q['id']):
        traces=ranker.rank(q['text'],limit=len(ranker.corpus.datasets))
        ids=set()
        for system in SYSTEMS:
            ids.update(t['dataset_id'] for t in sorted(traces,key=lambda t:(-t['scores'][system],t['dataset_id']))[:depth])
        if semantic is not None:
            if semantic.corpus.hash != ranker.corpus.hash:raise ValueError('Pool corpus mismatch')
            ids.update(t['dataset_id'] for t in semantic.rank(q['text'],limit=depth))
        if not ids: raise ValueError(f"No candidates for query {q['id']}; change query or include semantic retrieval")
        ids=sorted(ids);rng.shuffle(ids)
        records.append({'id':q['id'],'text':q['text'],'candidate_order':ids,'judgments':{i:None for i in ids}})
    result={'status':'inherited_unverified','provenance':{'kind':'unassessed_pool','corpus_hash':ranker.corpus.hash,'config_hash':ranker.policy_hash,'systems':list(SYSTEMS),'depth':depth,'seed':seed,'instructions':'Have independent assessors assign integer 0/1/2. Retain assessor records and adjudication separately; mark verified only after evidence review.'},'queries':records}
    if semantic is not None:
        result['provenance']['systems'].append('semantic')
        result['provenance']['semantic']=semantic.metadata
    if output is not None:write_json(output,result)
    return result
