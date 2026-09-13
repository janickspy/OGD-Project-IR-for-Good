"""Strict judgments and paired, explicitly pooled evaluation."""
import math
from itertools import combinations
import numpy as np
from scipy.stats import wilcoxon
from .io import digest, write_json

SYSTEMS = ('bm25', 'linear', 'mamdani', 'hybrid')
METRICS = ('map10', 'p5', 'ndcg10', 'mrr10')

def validate_judgments(data, corpus):
    if data.get('status') not in ('inherited_unverified', 'verified') or not data.get('provenance'):
        raise ValueError('Judgments require explicit status and provenance')
    queries = data['queries']
    if not queries or len({q['id'] for q in queries}) != len(queries):
        raise ValueError('Empty or duplicate queries')
    for q in queries:
        if not q['text'].strip() or not q['judgments']:
            raise ValueError('Empty query or judgments')
        for identity, grade in q['judgments'].items():
            if identity not in corpus.by_id or type(grade) is not int or grade not in (0, 1, 2):
                raise ValueError('Unknown dataset or missing/invalid grade; blank is not zero')
    return data

def import_judgments(data, corpus, output):
    validate_judgments(data, corpus)
    write_json(output, data)

def metrics(order, judgments):
    if len(set(order)) != len(order) or any(i not in judgments for i in order):
        raise ValueError('Duplicate result or unjudged result: obtain judgments first')
    grades = [judgments[i] for i in order[:10]]
    positives = sum(g > 0 for g in judgments.values())
    hits = 0; ap = 0.; rr = 0.
    for rank, grade in enumerate(grades, 1):
        if grade > 0:
            hits += 1; ap += hits / rank
            if not rr: rr = 1 / rank
    dcg = sum((2**g-1)/math.log2(r+2) for r,g in enumerate(grades))
    ideal = sum((2**g-1)/math.log2(r+2) for r,g in enumerate(sorted(judgments.values(), reverse=True)[:10]))
    return dict(map10=ap/positives if positives else 0., p5=sum(g>0 for g in grades[:5])/5,
                ndcg10=dcg/ideal if ideal else 0., mrr10=rr)

def evaluate(ranker, data, *, allow_unverified=False, mode='judged_pool', semantic=None):
    validate_judgments(data, ranker.corpus)
    if data['status'] != 'verified' and not allow_unverified:
        raise ValueError('Historical labels are unverified; explicitly opt into legacy reanalysis')
    if mode not in ('judged_pool', 'full_corpus'): raise ValueError('Unknown evaluation mode')
    if semantic is not None and semantic.corpus.hash != ranker.corpus.hash:
        raise ValueError('Semantic and lexical corpus hashes differ')
    systems = (*SYSTEMS, 'semantic') if semantic is not None else SYSTEMS
    rows = []; runs = {}
    for q in data['queries']:
        traces = ranker.rank(q['text'], candidates=q['judgments'] if mode=='judged_pool' else None,
                             limit=len(ranker.corpus.datasets), include_zero=mode=='judged_pool')
        runs[q['id']] = {}
        for system in systems:
            if system == 'semantic':
                order = [t['dataset_id'] for t in semantic.rank(q['text'], candidates=q['judgments'] if mode=='judged_pool' else None, limit=10)]
            else:
                order = [t['dataset_id'] for t in sorted(traces,key=lambda t:(-t['scores'][system], t['dataset_id']))][:10]
            runs[q['id']][system] = order
            rows.append(dict(query_id=q['id'],system=system,positive_count=sum(g>0 for g in q['judgments'].values()),
                             **metrics(order, q['judgments'])))
    aggregates = {}
    for subset in ('all_queries','positive_queries'):
        aggregates[subset] = {}
        for system in systems:
            selected = [r for r in rows if r['system']==system and (subset=='all_queries' or r['positive_count']>0)]
            aggregates[subset][system] = {'n':len(selected)}
            for metric in METRICS:
                vals = np.array([r[metric] for r in selected])
                rng = np.random.default_rng(2026)
                ci = np.quantile(rng.choice(vals,(2000,len(vals)),replace=True).mean(axis=1),[.025,.975]).tolist() if len(vals) else None
                aggregates[subset][system][metric] = {'mean':float(vals.mean()) if len(vals) else None,'bootstrap95':ci}
    # One prespecified family: all six pairwise comparisons for nDCG@10, all queries.
    tests = []
    for a,b in combinations(systems,2):
        va=np.array([r['ndcg10'] for r in rows if r['system']==a]); vb=np.array([r['ndcg10'] for r in rows if r['system']==b])
        delta=va-vb
        p=float(wilcoxon(delta, method='approx', zero_method='wilcox').pvalue) if np.any(delta) else 1.
        ci=np.quantile(np.random.default_rng(2026).choice(delta,(2000,len(delta)),replace=True).mean(axis=1),[.025,.975]).tolist()
        tests.append({'a':a,'b':b,'mean_difference':float(delta.mean()),'paired_bootstrap95':ci,'p':p})
    previous=0.
    for index,t in enumerate(sorted(tests,key=lambda t:t['p'])):
        previous=max(previous,min(1.,(len(tests)-index)*t['p'])); t['p_holm']=previous
    return {'schema_version':1,'mode':mode,'label_status':data['status'], 'config':ranker.config,
            'config_hash':ranker.policy_hash,'corpus_hash':ranker.corpus.hash,'judgments_hash':digest(data),
            'metric_policy':'AP@10 denominator: all known positive judgments; binary grade>0; nDCG gain 2^grade-1; P@5 denominator 5; unjudged results rejected',
            'uncertainty':'2000 query bootstrap samples, seed 2026; descriptive conditional on this small judgment pool',
            'tests_policy':f'{len(tests)} paired two-sided Wilcoxon normal approximations, zero differences removed; Holm family nDCG@10/all queries. Exploratory, not equivalence tests.',
            'systems':list(systems),'semantic':semantic.metadata if semantic is not None else None,
            'aggregates':aggregates,'paired_tests':tests,'per_query':rows,'runs':runs}
