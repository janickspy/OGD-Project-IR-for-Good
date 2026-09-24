"""Print every number reported in the paper, computed from the released data and results."""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ogd_ir.config import load_config
from ogd_ir.model import Corpus
from ogd_ir.ranking import Ranker, triangle

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
R = lambda name: json.loads((HERE / 'results' / name).read_text())


def main():
    corpus = Corpus.load(ROOT / 'data/legacy/corpus_v2.json')
    config = load_config(ROOT / 'configs/default.json')
    ranker = Ranker(corpus, config)
    queries = json.loads((ROOT / 'data/legacy/judgments.json').read_text())['queries']
    inherited = {q['id']: q['judgments'] for q in queries}
    report = json.loads((ROOT / 'results/application_reanalysis.json').read_text())
    yearbook = lambda d: 'Jahrbuch' in corpus.by_id[d].title
    out = {}

    # Collection and inherited judgments (Sect. 4.1)
    out['publishers'] = dict(Counter(d.publisher for d in corpus.datasets))
    out['yearbooks'] = sum(yearbook(d.id) for d in corpus.datasets)
    out['distinct_datasets_in_candidate_sets'] = len({d for j in inherited.values() for d in j})
    out['inherited_grades'] = dict(Counter(g for j in inherited.values() for g in j.values()))
    out['sets_without_positive'] = [q for q, j in inherited.items() if not any(g > 0 for g in j.values())]
    gate = {q['id']: sum(ranker.index.score(q['text'], d.id)[0] > 0 for d in corpus.datasets) for q in queries}
    out['topics_gate_admits_481_or_more'] = sum(v >= 481 for v in gate.values())

    # Table 2 and Sect. 5.1
    agg = report['aggregates']
    out['table2'] = {s: {m: round(agg['all_queries'][s][m]['mean'], 4) for m in ('map10', 'p5', 'ndcg10', 'mrr10')}
                     | {'ndcg10_positive': round(agg['positive_queries'][s]['ndcg10']['mean'], 4)} for s in agg['all_queries']}
    nd = {s: agg['all_queries'][s]['ndcg10']['mean'] for s in agg['all_queries']}
    out['mamdani_minus_bm25'] = round(nd['mamdani'] - nd['bm25'], 4)
    out['hybrid_minus_bm25'] = round(nd['hybrid'] - nd['bm25'], 4)
    out['hybrid_relative_cost'] = round(1 - nd['hybrid'] / nd['bm25'], 3)
    out['bootstrap95'] = {s: [round(x, 3) for x in agg['all_queries'][s]['ndcg10']['bootstrap95']] for s in ('semantic', 'hybrid')}
    tests = R('exact_paired_tests.json')['tests']
    out['smallest_holm_pooled'] = round(min(t['p_holm'] for t in tests), 4)
    out['nonzero_topic_differences'] = sorted({t['nonzero_query_pairs'] for t in tests})

    # Agreement and Table 3
    ja = R('judgment_analysis.json')
    out['agreement'] = {k: {'exact': round(v['exact'], 3), 'kappa_w': round(v['weighted_kappa_quadratic'], 2),
                            'binary_kappa': round(v['binary_kappa'], 2), 'zero_two': v['confusion'][0][2] + v['confusion'][2][0]}
                        for k, v in ja['agreement'].items()}
    out['table3'] = {s: {m: round(v[m]['mean'], 4) for m in ('ndcg10', 'p5', 'map10')}
                     for k in ('A', 'B') for s, v in [(f'{k}:{s}', v) for s, v in ja['full_corpus'][k]['summary'].items()]}
    out['full_tests_holm_below_0.05'] = {k: [(t['a'], t['b'], round(t['p_holm'], 3)) for t in ja['full_corpus'][k]['tests'] if t['p_holm'] < .05]
                                         for k in ('A', 'B')}
    for variant in ('A_grade2_only', 'B_grade2_only', 'A_yearbooks_not_relevant', 'B_yearbooks_not_relevant'):
        v = ja['full_corpus'][variant]
        out[variant] = {'ndcg10': {s: round(x['ndcg10']['mean'], 3) for s, x in v['summary'].items()},
                        'topics_with_relevant': v['topics_with_relevant']}
    out['pooled_llm_order'] = {k: ja['system_orderings'][f'pooled_{k}'] for k in ('A', 'B', 'inherited')}
    out['semantic_lead_over_bm25'] = {
        f'{setting}_{k}': round(ja[block][k]['summary']['semantic']['ndcg10']['mean'] - ja[block][k]['summary']['bm25']['ndcg10']['mean'], 3)
        for setting, block in (('pooled', 'pooled_reranking'), ('full', 'full_corpus')) for k in ('A', 'B')}

    # Extended pools (end of Sect. 5.1)
    runs = R('full_corpus_runs.json')['runs']
    grades = {k: {(j['query_id'], j['dataset_id']): j['grade']
                  for j in json.loads((HERE / f'judgments/assessor_{k}.json').read_text())['judgments']} for k in 'AB'}
    out['topics_with_positive_full_pool'] = {k: len({q for (q, d), g in grades[k].items() if g > 0}) for k in 'AB'}
    env01 = [d for (q, d), g in grades['A'].items() if q == 'ENV-01' and g == 2 and grades['B'][(q, d)] == 2]
    out['env01_relevant_both'] = len(env01)
    out['env01_in_inherited'] = sum(d in inherited['ENV-01'] for d in env01)
    out['env01_in_top10'] = {s: sum(d in runs['ENV-01'][s] for d in env01) for s in ('bm25', 'semantic')}
    out['grade1_yearbooks'] = {k: (sum(yearbook(d) for (q, d), g in grades[k].items() if g == 1),
                                   sum(1 for g in grades[k].values() if g == 1)) for k in 'AB'}
    out['yearbook_top10_positions'] = {s: sum(yearbook(d) for q in runs for d in runs[q][s]) for s in ('bm25', 'semantic')}
    topics = {t['id']: t['query'] for t in json.loads((HERE / 'judgments/topics.json').read_text())['topics']}
    share, total = Counter(), 0.
    for q in runs:
        for d in runs[q]['bm25']:
            if yearbook(d):
                for m in ranker.score(topics[q], d)['evidence']['matched_terms']:
                    share[m['term']] += m['bm25_contribution']
                    total += m['bm25_contribution']
    out['yearbook_bm25_share_schweiz'] = round(share['schweiz'] / total, 3)

    # Sect. 5.2
    fd = R('feature_diagnostics.json')
    out['modified_dates'] = fd['modified_dates']
    out['T_range'] = (round(fd['features']['freshness']['min'], 6), round(fd['features']['freshness']['max'], 6))
    out['min_high_membership_T'] = round(min((x['freshness'] - .5) / .5 for x in fd['rows']), 4)
    out['completeness_counts'] = fd['completeness_counts']
    out['A_values'] = sorted({round(x['resources'], 4) for x in fd['rows']})
    counts = [d.resource_count for d in corpus.datasets]
    out['resource_counts'] = (min(counts), max(counts))
    out['yearbook_resource_counts'] = dict(Counter(d.resource_count for d in corpus.datasets if yearbook(d.id)))
    out['other_resource_median'] = float(np.median([d.resource_count for d in corpus.datasets if not yearbook(d.id)]))
    fi = R('feature_interventions.json')
    out['interventions'] = [(v['intervention'], v['feature'], round(v['fixed_value'], 6), round(v['mean_ndcg10'], 4), v['changed_queries'])
                            for v in fi['variants']]
    median_a = next(v['fixed_value'] for v in fi['variants'] if v['feature'] == 'resources' and v['intervention'] == 'corpus_median')
    gains = {}
    base_runs = report['runs']
    for v in fi['variants']:
        if v['feature'] == 'resources' and v['intervention'] == 'corpus_median':
            base = {r['query_id']: r['ndcg10'] for r in report['per_query'] if r['system'] == 'hybrid'}
            gains = {r['query_id']: round(r['ndcg10'] - base[r['query_id']], 3) for r in v['per_query'] if abs(r['ndcg10'] - base[r['query_id']]) > 1e-12}
            xd01 = next(r['order'] for r in v['per_query'] if r['query_id'] == 'XD-01')
            promoted = [(corpus.by_id[d].title[:40], inherited['XD-01'][d], base_runs['XD-01']['hybrid'].index(d) + 1, xd01.index(d) + 1)
                        for d in xd01 if yearbook(d) and inherited['XD-01'][d] == 2]
    out['fixing_A_topic_gains'] = gains
    out['fixing_A_promoted_yearbooks_XD01'] = promoted

    # Sect. 5.3 and 5.4
    tv = R('trace_validation.json')
    out['max_centroid_error'] = tv['max_absolute_centroid_error']
    out['trace_consistency'] = R('trace_consistency.json')
    rc = R('resource_counterfactuals.json')
    out['resource_decreases'] = (rc['decreases'], rc['tested_pairs'], sorted({e['query_id'] for e in rc['examples']}),
                                 sorted({e['resources_before'] for e in rc['examples']}))
    worst = min(rc['examples'], key=lambda e: e['delta'])
    out['largest_decrease'] = (round(worst['score_before'], 6), round(worst['score_after'], 6))
    mem = config['memberships']['resources']
    bound = lambda a: max(float(triangle(a, mem[k])) for k in ('low', 'medium', 'high'))
    a5, a6 = math.log1p(5) / math.log1p(10), math.log1p(6) / math.log1p(10)
    out['bound_5_to_6_resources'] = (round(a5, 4), round(bound(a5), 3), round(bound(a6), 3))
    out['S_of_decreasing_records'] = sorted({round(ranker.score(topics[e['query_id']], e['dataset_id'])['features']['similarity'], 3)
                                            for e in rc['examples']})
    pc = R('policy_checks.json')
    out['local_variants'] = (sum(v['changed_queries'] == 0 for v in pc['sensitivity']),
                             round(min(v['mean_tau'] for v in pc['sensitivity']), 4),
                             round(min(v['ndcg10'] for v in pc['sensitivity']), 4), round(max(v['ndcg10'] for v in pc['sensitivity']), 4))
    out['grid'] = R('monotonicity_grid.json')['grid_points']

    # Sect. 5.5
    out['exposure_candidate_sets'] = {k: {p: round(100 * x, 2) for p, x in v.items()} for k, v in pc['exposure'].items()}
    out['exposure_full_collection'] = {k: {p: round(100 * x, 1) for p, x in v.items()} for k, v in pc['full_corpus_exposure'].items()}
    out['candidate_sets_cantonal_share'] = round(sum(corpus.by_id[d].publisher == 'kanton-thurgau'
                                                     for j in inherited.values() for d in j) / 150, 3)
    out['judged_top10_positions'] = {s: sum(d in inherited[q] for q in runs for d in runs[q][s]) for s in runs['ENV-01']}
    out['pool_pairs'] = (ja['pairs'], ja['pairs'] - ja['inherited_pairs'])
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
