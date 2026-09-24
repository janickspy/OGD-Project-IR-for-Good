"""Audit checks C4 (local policy variants) and C5 (publisher exposure) on the
normalised corpus, using the inherited ten-record pools."""
import json
import math
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / 'src'))
from ogd_ir.config import FEATURES, load_config
from ogd_ir.evaluation import evaluate
from ogd_ir.model import Corpus
from ogd_ir.ranking import Ranker, rules


def variants(config):
    out = []
    for feature in FEATURES:
        for delta in (-.05, .05):
            v = deepcopy(config)
            v['memberships'][feature]['medium'][1] += delta
            out.append((f'{feature} middle peak {delta:+.2f}', v))
    for value in (.5, .8):
        v = deepcopy(config)
        v['alpha'] = value
        out.append((f'alpha {value}', v))
    for term in ('low', 'moderate', 'good', 'excellent'):
        v = deepcopy(config)
        v['rule_weights'] = {r['id']: .9 for r in rules(v) if r['then'] == term}
        out.append((f'{term} rule weights 0.9', v))
    return out


def _by_system(runs):
    out = {}
    for per_query in runs.values():
        for system, order in per_query.items():
            out.setdefault(system, []).append(order)
    return out


def main():
    corpus = Corpus.load(REPO / 'data/legacy/corpus_v2.json')
    judgments = json.loads((REPO / 'data/legacy/judgments.json').read_text())
    config = load_config(REPO / 'configs/default.json')
    baseline = evaluate(Ranker(corpus, config), judgments, allow_unverified=True)

    exposure = {}
    for system in ('bm25', 'linear', 'mamdani', 'hybrid'):
        counts = Counter()
        for runs in baseline['runs'].values():
            for i, identity in enumerate(runs[system]):
                counts[corpus.by_id[identity].publisher] += 1 / math.log2(i + 2)
        total = sum(counts.values())
        exposure[system] = {p: v / total for p, v in sorted(counts.items())}

    # Attribution: the hybrid with the resource feature held at its corpus median.
    ranker = Ranker(corpus, config)
    median_a = float(np.median([ranker.features('', d)[0]['resources'] for d in corpus.datasets]))
    counts = Counter()
    for q in judgments['queries']:
        traces = [ranker.score(q['text'], d) for d in q['judgments']]
        def fixed_a(t):
            values = dict(t['features'], resources=median_a)
            fuzzy, _ = ranker.infer(values)
            h = config['alpha'] * fuzzy + (1 - config['alpha']) * sum(values.values()) / len(values)
            return h if t['lexical_gate_passed'] else 0.
        order = [t['dataset_id'] for t in sorted(traces, key=lambda t: (-fixed_a(t), t['dataset_id']))]
        for i, identity in enumerate(order):
            counts[corpus.by_id[identity].publisher] += 1 / math.log2(i + 2)
    total = sum(counts.values())
    exposure['hybrid_resources_fixed_at_median'] = {p: v / total for p, v in sorted(counts.items())}

    full_runs = ROOT / 'results/full_corpus_runs.json'
    full_exposure = {}
    if full_runs.exists():
        for system, lists in _by_system(json.loads(full_runs.read_text())['runs']).items():
            counts = Counter()
            for order in lists:
                for i, identity in enumerate(order):
                    counts[corpus.by_id[identity].publisher] += 1 / math.log2(i + 2)
            total = sum(counts.values())
            full_exposure[system] = {p: v / total for p, v in sorted(counts.items())}

    sensitivity = []
    for name, variant in variants(config):
        result = evaluate(Ranker(corpus, variant), judgments, allow_unverified=True)
        taus, changed = [], 0
        for qid, runs in baseline['runs'].items():
            a, b = runs['hybrid'], result['runs'][qid]['hybrid']
            taus.append(float(kendalltau(list(range(len(a))), [b.index(x) for x in a]).statistic))
            changed += a != b
        sensitivity.append({'name': name, 'changed_queries': changed, 'mean_tau': float(np.mean(taus)),
                            'ndcg10': result['aggregates']['all_queries']['hybrid']['ndcg10']['mean']})

    out = {'corpus_hash': corpus.hash, 'config_hash': baseline['config_hash'],
           'exposure_scope': 'Share of 1/log2(rank+1) exposure by publisher over the 15 pooled rankings.',
           'exposure': exposure,
           'full_corpus_exposure_scope': 'Same measure over the full-collection top-ten runs (results/full_corpus_runs.json).',
           'full_corpus_exposure': full_exposure, 'sensitivity': sensitivity}
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/policy_checks.json').write_text(json.dumps(out, indent=2) + '\n')
    for system in ('bm25', 'hybrid', 'hybrid_resources_fixed_at_median'):
        print(system, {p: round(100 * v, 2) for p, v in exposure[system].items()})
    for system, shares in full_exposure.items():
        print('full collection', system, {p: round(100 * v, 1) for p, v in shares.items()})
    print('variants preserving all orders:', sum(v['changed_queries'] == 0 for v in sensitivity),
          'tau range:', round(min(v['mean_tau'] for v in sensitivity), 4), round(max(v['mean_tau'] for v in sensitivity), 4),
          'nDCG@10 range:', round(min(v['ndcg10'] for v in sensitivity), 4), round(max(v['ndcg10'] for v in sensitivity), 4))


if __name__ == '__main__':
    main()
