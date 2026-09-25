"""BM25 with German stopword removal, Snowball stemming and compound splitting.

A stronger lexical baseline than the application's BM25. It keeps the same
fields, field weights, k1 and b, but removes German stopwords (Snowball list),
stems every token with the Snowball German stemmer and, in the `split` variant,
also indexes the parts of compounds with at least eight letters when the
CharSplit model (package compound-split) scores the split at 0.5 or higher.
The original token is kept, so splitting only adds evidence.

Writes results/german_bm25_runs.json with candidate-set and full-collection
top-ten runs for both variants.
"""
import json
import math
from collections import Counter
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import snowballstemmer

from ogd_ir.config import load_config
from ogd_ir.model import Corpus, tokens

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
STOPWORDS = {w.strip() for w in (HERE / 'german_stopwords.txt').read_text(encoding='utf-8').splitlines()
             if w.strip() and not w.startswith('#')}
STEMMER = snowballstemmer.stemmer('german')
MIN_LENGTH = 8
MIN_SCORE = 0.5
VARIANTS = {'bm25_stem': False, 'bm25_split': True}


@lru_cache(maxsize=None)
def parts(word, depth=2):
    """Compound parts of a word, or the word itself if no confident split exists."""
    from compound_split import char_split
    if depth == 0 or len(word) < MIN_LENGTH or not word.isalpha():
        return (word,)
    score, head, tail = char_split.split_compound(word)[0]
    if score < MIN_SCORE:
        return (word,)
    return parts(head.lower(), depth - 1) + parts(tail.lower(), depth - 1)


@lru_cache(maxsize=None)
def analyse_token(token, split):
    if token in STOPWORDS:
        return ()
    terms = [token]
    if split:
        pieces = parts(token)
        if len(pieces) > 1:
            terms += [p for p in pieces if p not in STOPWORDS]
    return tuple(STEMMER.stemWord(t) for t in terms)


def analyse(text, split):
    out = []
    for token in tokens(text):
        out.extend(analyse_token(token, split))
    return out


class GermanBM25:
    def __init__(self, corpus, config, split):
        self.split = split
        self.k1, self.b = config['bm25']['k1'], config['bm25']['b']
        self.docs = {}
        for d in corpus.datasets:
            counts = Counter()
            for field, weight in config['bm25']['field_weights'].items():
                for term, n in Counter(analyse(getattr(d, field), split)).items():
                    counts[term] += n * weight
            self.docs[d.id] = counts
        self.df = Counter()
        for counts in self.docs.values():
            self.df.update(counts.keys())
        self.lengths = {k: sum(v.values()) for k, v in self.docs.items()}
        self.avgdl = sum(self.lengths.values()) / len(self.docs)

    def score(self, query, identity):
        counts, n = self.docs[identity], len(self.docs)
        total = 0.
        for term in sorted(set(analyse(query, self.split))):
            tf = counts.get(term, 0)
            if tf:
                idf = math.log1p((n - self.df[term] + .5) / (self.df[term] + .5))
                total += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[identity] / self.avgdl))
        return total

    def rank(self, query, candidates=None, limit=10):
        ids = sorted(candidates) if candidates is not None else sorted(self.docs)
        scored = [(self.score(query, i), i) for i in ids]
        if candidates is None:
            scored = [x for x in scored if x[0] > 0]
        return [i for s, i in sorted(scored, key=lambda x: (-x[0], x[1]))[:limit]]


def main():
    corpus = Corpus.load(ROOT / 'data/legacy/corpus_v2.json')
    config = load_config(ROOT / 'configs/default.json')
    queries = json.loads((ROOT / 'data/legacy/judgments.json').read_text())['queries']
    out = {'corpus_hash': corpus.hash, 'stopwords': 'Snowball German list (audit/german_stopwords.txt)',
           'stemmer': f"snowballstemmer {version('snowballstemmer')} (German)",
           'compound_splitting': {'package': f"compound-split {version('compound-split')} (CharSplit model)",
                                  'min_length': MIN_LENGTH, 'min_score': MIN_SCORE, 'depth': 2, 'original_token_kept': True},
           'bm25': config['bm25'], 'candidate_sets': {}, 'full_collection': {}}
    for name, split in VARIANTS.items():
        index = GermanBM25(corpus, config, split)
        out['candidate_sets'][name] = {q['id']: index.rank(q['text'], candidates=q['judgments']) for q in queries}
        out['full_collection'][name] = {q['id']: index.rank(q['text']) for q in queries}
        print(name, 'query terms:', {q['id']: analyse(q['text'], split) for q in queries[:2]})
    (HERE / 'results/german_bm25_runs.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
