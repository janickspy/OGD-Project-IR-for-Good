"""Additional C1 checks on the pooled traces.

1. Each trace's aggregated output strengths equal the maximum firing strength
   of its recorded active rules per consequent, and the centroid recomputed
   from those strengths equals the recorded Mamdani score.
2. The hybrid score equals the declared mixture of the recorded components.
3. With --model-cache, the pooled semantic orders are recomputed with the
   pinned model and compared with the application report.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ogd_ir.config import load_config
from ogd_ir.model import Corpus
from ogd_ir.ranking import triangle

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-cache')
    args = p.parse_args()
    config = load_config(ROOT / 'configs/default.json')
    traces = json.loads((HERE / 'results/pooled_traces.json').read_text())
    universe = np.linspace(0., 1., config['universe_points'])
    curves = {term: triangle(universe, points) for term, points in config['output_memberships'].items()}
    strength_error = centroid_error = hybrid_error = 0.
    for t in traces:
        inference = t['inference']
        strengths = {term: 0. for term in curves}
        for rule in inference['active_rules']:
            strengths[rule['then']] = max(strengths[rule['then']], rule['strength'])
        strength_error = max(strength_error, max(abs(strengths[k] - inference['output_strengths'][k]) for k in curves))
        curve = np.maximum.reduce([np.minimum(curves[k], s) for k, s in strengths.items()])
        centroid = float(np.dot(universe, curve) / curve.sum())
        centroid_error = max(centroid_error, abs(centroid - inference['centroid']))
        linear = sum(t['linear_contributions'].values())
        hybrid = t['alpha'] * inference['centroid'] + (1 - t['alpha']) * linear
        hybrid_error = max(hybrid_error, abs(hybrid - t['ungated_hybrid']))
    out = {'traces': len(traces), 'max_strength_error': strength_error,
           'max_centroid_error_from_recorded_rules': centroid_error, 'max_hybrid_error': hybrid_error}

    if args.model_cache:
        from ogd_ir.semantic import MiniLMEncoder, SemanticIndex
        corpus = Corpus.load(ROOT / 'data/legacy/corpus_v2.json')
        index = SemanticIndex(corpus, MiniLMEncoder(args.model_cache, offline=True))
        report = json.loads((ROOT / 'results/application_reanalysis.json').read_text())
        queries = json.loads((ROOT / 'data/legacy/judgments.json').read_text())['queries']
        same = sum([r['dataset_id'] for r in index.rank(q['text'], candidates=q['judgments'], limit=10)]
                   == report['runs'][q['id']]['semantic'] for q in queries)
        out['semantic_pooled_orders_reproduced'] = f'{same}/{len(queries)}'
    (HERE / 'results/trace_consistency.json').write_text(json.dumps(out, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
