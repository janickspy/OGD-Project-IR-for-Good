"""Audit check C4 at observed inputs: add one resource to each inherited
query-record pair with all other features fixed, and trace the hybrid score
of the record with the largest decrease over resource counts 3 to 15."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / 'src'))
from ogd_ir.model import Corpus
from ogd_ir.ranking import Ranker


def resources(count, cap):
    return min(math.log1p(count) / math.log1p(cap), 1.)


def hybrid(ranker, values):
    fuzzy, _ = ranker.infer(values)
    linear = sum(values.values()) / len(values)
    return ranker.config['alpha'] * fuzzy + (1 - ranker.config['alpha']) * linear, fuzzy, linear


def main():
    corpus = Corpus.load(REPO / 'data/legacy/corpus_v2.json')
    ranker = Ranker(corpus)
    cap = ranker.config['resource_cap']
    queries = json.loads((REPO / 'data/legacy/judgments.json').read_text())['queries']
    decreases, tested = [], 0
    for q in queries:
        for identifier in q['judgments']:
            trace = ranker.score(q['text'], identifier)
            if not trace['lexical_gate_passed']:
                continue
            tested += 1
            count = trace['evidence']['resource_count']
            after, _, _ = hybrid(ranker, dict(trace['features'], resources=resources(count + 1, cap)))
            before = trace['scores']['hybrid']
            if after < before - 1e-10:
                decreases.append({'query_id': q['id'], 'dataset_id': identifier, 'resources_before': count,
                                  'score_before': before, 'score_after': after, 'delta': after - before})
    worst = min(decreases, key=lambda d: d['delta'])
    text = next(q['text'] for q in queries if q['id'] == worst['query_id'])
    features = ranker.score(text, worst['dataset_id'])['features']
    rows = []
    for count in range(3, 16):
        values = dict(features, resources=resources(count, cap))
        h, f, l = hybrid(ranker, values)
        rows.append({'resources': count, 'A': values['resources'], 'mamdani': f, 'linear': l, 'hybrid': h})
    out = {'intervention': 'One additional resource, other feature values held fixed', 'tested_pairs': tested,
           'decreases': len(decreases), 'examples': decreases,
           'figure': {'query_id': worst['query_id'], 'dataset_id': worst['dataset_id'],
                      'title': corpus.by_id[worst['dataset_id']].title, 'fixed_features': features, 'rows': rows},
           'config_hash': ranker.policy_hash, 'corpus_hash': corpus.hash}
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/resource_counterfactuals.json').write_text(json.dumps(out, indent=2) + '\n')
    print(f'{len(decreases)} decreases among {tested} pairs; largest {worst["score_before"]:.6f} -> {worst["score_after"]:.6f}')

    try:
        import matplotlib
    except ImportError:
        print('matplotlib not installed; figure skipped')
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(5.9, 2.4), layout='constrained')
    ax.plot([r['resources'] for r in rows], [r['hybrid'] for r in rows], color='#185b70',
            marker='o', linewidth=1.5, markersize=3.5)
    before = next(r for r in rows if r['resources'] == worst['resources_before'])
    after = next(r for r in rows if r['resources'] == worst['resources_before'] + 1)
    ax.scatter([before['resources'], after['resources']], [before['hybrid'], after['hybrid']], color='#a34230', zorder=3)
    ax.annotate(f"{before['resources']} → {after['resources']} resources\n{before['hybrid']:.6f} → {after['hybrid']:.6f}",
                xy=(after['resources'], after['hybrid']), xytext=(9.1, .548), fontsize=8,
                arrowprops={'arrowstyle': '->', 'color': '#a34230', 'linewidth': .8})
    ax.set(xlabel='Hypothetical resource count (other features fixed)', ylabel='Hybrid score',
           xticks=range(3, 16), xlim=(2.7, 15.3))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.grid(axis='y', color='#dddddd', linewidth=.5)
    fig.savefig(ROOT / 'results/resource_counterfactual.pdf', metadata={'CreationDate': None})


if __name__ == '__main__':
    main()
