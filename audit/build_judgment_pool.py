"""Build the depth-10 judgment pool for the full-corpus evaluation.

The pool is the union of the inherited ten-record pools and the top ten
results of every system over the whole corpus. Each assessor receives the
same items grouped by topic, in an independently shuffled order, without
system names, ranks, scores or existing labels.

With --round2, the pool is extended by the top ten results of the German BM25
baselines (results/german_bm25_runs.json) that are not yet judged; these pairs
are written to judgments/round2/ with their own shuffling seeds.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from ogd_ir.config import load_config
from ogd_ir.model import Corpus
from ogd_ir.ranking import Ranker

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
LEXICAL = ['bm25', 'linear', 'mamdani', 'hybrid']
THEMES = {'gove': 'Government and public sector', 'soci': 'Population and society',
          'econ': 'Economy and finance', 'educ': 'Education, culture and sport',
          'envi': 'Environment', 'regi': 'Regions and cities', 'tran': 'Transport',
          'heal': 'Health', 'agri': 'Agriculture, fisheries, forestry and food',
          'tech': 'Science and technology', 'ener': 'Energy', 'just': 'Justice and public safety',
          'intr': 'International issues'}
DOMAINS = {'ENV': 'environment', 'MOB': 'mobility', 'XD': 'cross-domain'}


def full_corpus_runs(ranker, topics, model_cache):
    runs = {}
    semantic = None
    if model_cache:
        from ogd_ir.semantic import MiniLMEncoder, SemanticIndex
        semantic = SemanticIndex(ranker.corpus, MiniLMEncoder(model_cache, offline=True))
    for topic in topics:
        runs[topic['id']] = {s: [t['dataset_id'] for t in ranker.rank(topic['query'], s, limit=10)] for s in LEXICAL}
        if semantic is not None:
            runs[topic['id']]['semantic'] = [t['dataset_id'] for t in semantic.rank(topic['query'], limit=10)]
    return runs


def item_id(query_id, dataset_id):
    return 'i' + hashlib.sha256(f'{query_id}|{dataset_id}'.encode()).hexdigest()[:8]


def text(value, limit=None):
    if isinstance(value, dict):
        parts = []
        for lang in ('de', 'fr', 'it', 'en'):
            v = (value.get(lang) or '').strip()
            if v and v not in parts:
                parts.append(v)
        value = ' / '.join(parts)
    value = ' '.join(str(value or '').split())
    return value if limit is None or len(value) <= limit else value[:limit].rsplit(' ', 1)[0] + ' [...]'


def render(record):
    frequency = (record.get('frequency') or '').rsplit('/', 1)[-1]
    lines = [f"Title: {text(record['title'])}",
             f"Publisher: {text(record['publisher'])}",
             f"Description: {text(record['description'], 1500) or '(none)'}",
             f"Keywords: {', '.join(record.get('keywords', [])[:20]) or '(none)'}",
             f"Themes: {', '.join(THEMES.get(t, t) for t in record.get('themes', [])) or '(none)'}"]
    if record.get('spatial'):
        lines.append(f"Spatial coverage: {record['spatial']}")
    if frequency:
        lines.append(f"Update frequency: {frequency}")
    if record.get('formats'):
        lines.append(f"Formats: {', '.join(record['formats'])}")
    return lines


def write_batches(pairs, topics, catalog, out, seeds):
    for assessor, seed in seeds:
        rng = random.Random(seed)
        for prefix, domain in DOMAINS.items():
            lines = [f'# Assessment batch {assessor}-{prefix} ({domain} topics)', '']
            for topic in (t for t in topics if t['id'].startswith(prefix + '-')):
                items = [d for q, d in pairs if q == topic['id']]
                if not items:
                    continue
                rng.shuffle(items)
                lines += [f"## Topic {topic['id']}", f"Query: {topic['query']}",
                          f"Information need: {topic['description']}", f'Items: {len(items)}', '']
                for d in items:
                    lines += [f"### Item {item_id(topic['id'], d)}"] + render(catalog[d]) + ['']
            (out / f'batch_{assessor}_{prefix}.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-cache', help='pinned MiniLM cache; omit to reuse audit/results/full_corpus_runs.json')
    p.add_argument('--output', default=str(HERE / 'judgments'))
    p.add_argument('--round2', action='store_true', help='extend the pool with the German BM25 runs')
    args = p.parse_args()
    corpus = Corpus.load(ROOT / 'data/legacy/corpus_v2.json')
    catalog = json.loads((ROOT / 'data/legacy/catalog.json').read_text())
    inherited = json.loads((ROOT / 'data/legacy/judgments.json').read_text())
    topics = json.loads((HERE / 'judgments/topics.json').read_text())['topics']
    assert [t['query'] for t in topics] == [q['text'] for q in inherited['queries']]
    out = Path(args.output)

    if args.round2:
        judged = {(v['query_id'], v['dataset_id'])
                  for v in json.loads((out / 'pool_key.json').read_text())['items'].values()}
        german = json.loads((HERE / 'results/german_bm25_runs.json').read_text())['full_collection']
        pairs = sorted({(q, d) for runs in german.values() for q, ds in runs.items() for d in ds} - judged)
        out = out / 'round2'
        out.mkdir(parents=True, exist_ok=True)
        key = {item_id(q, d): {'query_id': q, 'dataset_id': d} for q, d in pairs}
        (out / 'pool_key.json').write_text(json.dumps({'pairs': len(pairs), 'items': key}, indent=2) + '\n')
        write_batches(pairs, topics, catalog, out, (('A', 3307), ('B', 4409)))
        print(f'{len(pairs)} additional pairs')
        return

    ranker = Ranker(corpus, load_config(ROOT / 'configs/default.json'))
    runs_path = HERE / 'results/full_corpus_runs.json'
    if args.model_cache:
        runs = full_corpus_runs(ranker, topics, args.model_cache)
        runs_path.write_text(json.dumps({'corpus_hash': corpus.hash, 'config_hash': ranker.policy_hash,
                                         'runs': runs}, indent=2) + '\n')
    runs = json.loads(runs_path.read_text())['runs']

    pairs = []
    for q in inherited['queries']:
        ids = set(q['judgments'])
        for system_run in runs[q['id']].values():
            ids.update(system_run)
        pairs.extend((q['id'], d) for d in sorted(ids))
    key = {item_id(q, d): {'query_id': q, 'dataset_id': d} for q, d in pairs}
    assert len(key) == len(pairs)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pool_key.json').write_text(json.dumps({'pairs': len(pairs), 'items': key}, indent=2) + '\n')
    write_batches(pairs, topics, catalog, out, (('A', 1101), ('B', 2203)))
    print(f'{len(pairs)} pairs; {sum(len(q["judgments"]) for q in inherited["queries"])} inherited')


if __name__ == '__main__':
    main()
