"""Agreement of the extended judgments and full-corpus effectiveness.

Reads the two assessor files in audit/judgments, compares them with each
other and with the inherited labels, and evaluates the full-corpus top-ten
runs of all five systems under each assessor's labels.
"""
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from ogd_ir.evaluation import metrics

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SYSTEMS = ['semantic', 'bm25', 'mamdani', 'hybrid', 'linear']
METRICS = ['map10', 'p5', 'ndcg10', 'mrr10']


def weighted_kappa(x, y, k=3):
    observed = np.zeros((k, k))
    for a, b in zip(x, y):
        observed[a, b] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(1), observed.sum(0))
    weights = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    return float(1 - (weights * observed).sum() / (weights * expected).sum())


def cohen_kappa(x, y):
    x, y = np.asarray(x), np.asarray(y)
    po = float(np.mean(x == y))
    pe = sum(float(np.mean(x == c) * np.mean(y == c)) for c in set(x) | set(y))
    return (po - pe) / (1 - pe) if pe < 1 else None


def agreement(x, y):
    return {'pairs': len(x), 'exact': float(np.mean(np.array(x) == np.array(y))),
            'weighted_kappa_quadratic': weighted_kappa(x, y),
            'binary_kappa': cohen_kappa([g > 0 for g in x], [g > 0 for g in y]),
            'confusion': [[sum(1 for a, b in zip(x, y) if a == i and b == j) for j in range(3)] for i in range(3)]}


def exact_permutation(values, a, b, queries):
    diffs = [values[a][q] - values[b][q] for q in queries]
    nonzero = [d for d in diffs if abs(d) >= 1e-12]
    observed = abs(np.mean(diffs))
    hits = total = 0
    for signs in itertools.product([1, -1], repeat=len(nonzero)):
        total += 1
        hits += abs(sum(s * d for s, d in zip(signs, nonzero)) / len(queries)) >= observed - 1e-12
    return {'a': a, 'b': b, 'mean_difference': float(np.mean(diffs)), 'nonzero_queries': len(nonzero), 'p': hits / total}


def holm(tests):
    previous = 0.
    for i, t in enumerate(sorted(tests, key=lambda t: t['p'])):
        previous = max(previous, min(1., (len(tests) - i) * t['p']))
        t['p_holm'] = previous
    return tests


def evaluate(runs, qrels):
    queries = sorted(qrels)
    per_query = {s: {q: metrics(runs[q][s], qrels[q]) for q in queries} for s in SYSTEMS}
    summary = {}
    for s in SYSTEMS:
        summary[s] = {}
        for m in METRICS:
            vals = np.array([per_query[s][q][m] for q in queries])
            boot = np.random.default_rng(2026).choice(vals, (2000, len(vals)), replace=True).mean(axis=1)
            summary[s][m] = {'mean': float(vals.mean()), 'bootstrap95': np.quantile(boot, [.025, .975]).tolist()}
    ndcg = {s: {q: per_query[s][q]['ndcg10'] for q in queries} for s in SYSTEMS}
    tests = holm([exact_permutation(ndcg, a, b, queries) for a, b in itertools.combinations(SYSTEMS, 2)])
    return {'summary': summary, 'tests': tests, 'per_query': per_query,
            'topics_with_relevant': sum(any(g > 0 for g in qrels[q].values()) for q in queries)}


def ordering(summary):
    return sorted(SYSTEMS, key=lambda s: -summary[s]['ndcg10']['mean'])


def main():
    inherited = {q['id']: q['judgments'] for q in json.loads((ROOT / 'data/legacy/judgments.json').read_text())['queries']}
    runs = json.loads((HERE / 'results/full_corpus_runs.json').read_text())['runs']
    pooled_runs = json.loads((ROOT / 'results/application_reanalysis.json').read_text())['runs']
    assessors = {}
    for name in 'AB':
        doc = json.loads((HERE / f'judgments/assessor_{name}.json').read_text())
        qrels = {}
        for j in doc['judgments']:
            qrels.setdefault(j['query_id'], {})[j['dataset_id']] = j['grade']
        assessors[name] = {'model': doc['model'], 'qrels': qrels}
    pairs = sorted((q, d) for q in assessors['A']['qrels'] for d in assessors['A']['qrels'][q])
    old = [(q, d) for q, d in pairs if d in inherited[q]]
    grade = lambda name, q, d: assessors[name]['qrels'][q][d]
    yearbook = json.loads((ROOT / 'data/legacy/catalog.json').read_text())
    not_yearbook = [(q, d) for q, d in old if 'Jahrbuch' not in yearbook[d]['title'].get('de', '')]

    coverage = {s: sum(d in inherited[q] for q in runs for d in runs[q][s]) for s in SYSTEMS}
    result = {'pairs': len(pairs), 'inherited_pairs': len(old),
              'inherited_coverage_of_full_corpus_top10': {'positions_per_system': 150, 'judged': coverage},
              'grade_counts': {n: dict(Counter(grade(n, q, d) for q, d in pairs)) for n in 'AB'},
              'grade_counts_inherited': dict(Counter(inherited[q][d] for q, d in old)),
              'agreement': {
                  'A_vs_inherited': agreement([grade('A', q, d) for q, d in old], [inherited[q][d] for q, d in old]),
                  'B_vs_inherited': agreement([grade('B', q, d) for q, d in old], [inherited[q][d] for q, d in old]),
                  'A_vs_B_all': agreement([grade('A', q, d) for q, d in pairs], [grade('B', q, d) for q, d in pairs]),
                  'A_vs_B_inherited_pairs': agreement([grade('A', q, d) for q, d in old], [grade('B', q, d) for q, d in old]),
                  'A_vs_B_new_pairs': agreement([grade('A', q, d) for q, d in pairs if d not in inherited[q]],
                                                [grade('B', q, d) for q, d in pairs if d not in inherited[q]]),
                  'A_vs_inherited_without_yearbooks': agreement([grade('A', q, d) for q, d in not_yearbook], [inherited[q][d] for q, d in not_yearbook]),
                  'B_vs_inherited_without_yearbooks': agreement([grade('B', q, d) for q, d in not_yearbook], [inherited[q][d] for q, d in not_yearbook])}}

    consensus = {q: {d: min(assessors['A']['qrels'][q][d], assessors['B']['qrels'][q][d]) for d in assessors['A']['qrels'][q]}
                 for q in assessors['A']['qrels']}
    strict = {name: {q: {d: 2 if g == 2 else 0 for d, g in assessors[name]['qrels'][q].items()}
                     for q in assessors[name]['qrels']} for name in 'AB'}
    # Sensitivity to general statistical yearbooks, which the guideline names as a grade-1 example.
    is_yearbook = lambda d: 'Jahrbuch' in yearbook[d]['title'].get('de', '')
    no_yearbooks = {name: {q: {d: 0 if is_yearbook(d) else g for d, g in assessors[name]['qrels'][q].items()}
                           for q in assessors[name]['qrels']} for name in 'AB'}
    full = {'A': evaluate(runs, assessors['A']['qrels']), 'B': evaluate(runs, assessors['B']['qrels']),
            'min_AB': evaluate(runs, consensus),
            'A_grade2_only': evaluate(runs, strict['A']), 'B_grade2_only': evaluate(runs, strict['B']),
            'A_yearbooks_not_relevant': evaluate(runs, no_yearbooks['A']),
            'B_yearbooks_not_relevant': evaluate(runs, no_yearbooks['B'])}
    pooled = {name: evaluate(pooled_runs, {q: {d: assessors[name]['qrels'][q][d] for d in inherited[q]} for q in inherited})
              for name in 'AB'}
    pooled['inherited'] = evaluate(pooled_runs, inherited)
    orders = {f'full_{k}': ordering(v['summary']) for k, v in full.items()}
    orders.update({f'pooled_{k}': ordering(v['summary']) for k, v in pooled.items()})
    rank_of = lambda order: [order.index(s) for s in SYSTEMS]
    taus = {f'{a}~{b}': float(kendalltau(rank_of(orders[a]), rank_of(orders[b])).statistic)
            for a, b in itertools.combinations(orders, 2)}
    result.update({'full_corpus': {k: {x: v[x] for x in ('summary', 'tests', 'topics_with_relevant')} for k, v in full.items()},
                   'pooled_reranking': {k: {x: v[x] for x in ('summary', 'tests', 'topics_with_relevant')} for k, v in pooled.items()},
                   'system_orderings': orders, 'kendall_tau_between_orderings': taus,
                   'per_query_full_corpus_ndcg10': {k: {s: {q: v['per_query'][s][q]['ndcg10'] for q in v['per_query'][s]}
                                                        for s in SYSTEMS} for k, v in full.items()}})
    (HERE / 'results/judgment_analysis.json').write_text(json.dumps(result, indent=2) + '\n')

    print('inherited labels in full-collection top ten:', coverage)
    ag = result['agreement']
    for k, v in ag.items():
        print(f"{k:34s} n={v['pairs']:3d} exact={v['exact']:.3f} qwk={v['weighted_kappa_quadratic']:.3f} binary_k={v['binary_kappa']:.3f}")
    for label, block in (('full', full), ('pooled', pooled)):
        for k, v in block.items():
            print(label, k, 'topics with relevant:', v['topics_with_relevant'],
                  {s: round(v['summary'][s]['ndcg10']['mean'], 4) for s in SYSTEMS})
    print(orders)
    print(taus)


if __name__ == '__main__':
    main()
