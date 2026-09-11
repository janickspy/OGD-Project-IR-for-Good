# OGD Project: Transparent Ranking for IR for Good

Creator and maintainer: **Janick Spycher**. This is a new implementation of transparent multi-criteria dataset ranking, for Swiss open-government datasets. It does not modify or reuse the predecessor application's source code. Deep Shukla's thesis and historical metadata/judgments are credited in [provenance](docs/PROVENANCE.md).

The engine, benchmark and explanations share one configuration. It provides corpus-wide BM25, a linear comparator, pure Mamdani inference and an explicitly disclosed hybrid; exact feature/rule/score traces; frozen-date evaluation; guarded annotation imports; uncertainty estimates; and reproducible sensitivity experiments.

**Tool status:** working offline ranking and evaluation engine with tested safeguards and a completed legacy reanalysis. The inherited judgments are unverified; results are conditional on those labels. See [limitations](docs/LIMITATIONS.md).

## Run

Python 3.11+:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
make test
ogd-ir --corpus data/legacy/corpus.json --config configs/default.json search 'Verkehrsunfälle Statistik Schweiz' --explain
make research
```

`make research` downloads checksum-verified historical inputs if needed, normalizes them and regenerates all results. No API key or live portal is used during ranking/evaluation. The example corpus is included for offline use.

## What is measured

[Results](results/legacy_reanalysis.json) are **reranking of inherited ten-document pools**, not end-to-end portal retrieval. All four systems use identical candidates and actual full-corpus BM25 statistics. The report separates all 15 queries from nine with positive judgments, logs every ranking, and specifies metrics and the six-comparison Holm family. Never compare these new numbers as a controlled improvement over the predecessor: multiple design choices changed.

```sh
ogd-ir --corpus data/legacy/corpus.json evaluate --judgments data/legacy/judgments.json --allow-unverified --output results/legacy_reanalysis.json
PYTHONPATH=src python scripts/create_annotation_pool.py
ogd-ir --corpus data/legacy/corpus.json import-judgments --input assessed.json --output data/local/verified.json
```

The generated annotation pool contains `null` labels. Import rejects blanks, unknown IDs, invalid grades and duplicate JSON keys before touching the output. A `verified` status records the researcher's attestation; software cannot authenticate an assessor. Retain independent source records. Full-corpus evaluation rejects unjudged returned results; known judgments still cannot prove exhaustive recall.

## Policy and explanations

`configs/default.json` is the complete policy, with `configs/rules.json` its 81 generated rules. These are declared policy choices, not fitted parameters. Similarity is saturated BM25 times query coverage. Completeness measures eight metadata-presence checks; freshness measures metadata age, not data currency; resource count is not verified availability. See [method](docs/METHOD.md).

Every JSON result exposes exact memberships, active rule strengths, centroid, linear contributions, blend, lexical gate, date handling and hashes. [Example](results/explanation_example.txt) is generated from the same scoring call used in evaluation. Ordinary search excludes zero lexical matches; pooled experiments retain them with zero final scores and deterministic ID tie-breaking.

## Tool documentation and results

- [Capabilities and limitations](docs/LIMITATIONS.md)
- [Evaluation and annotation workflow](docs/EVALUATION.md)
- [Sensitivity](results/sensitivity.json), [corpus audit](results/corpus_audit.json), [runtime manifest](results/environment.json)

MIT applies to the new software. Historical metadata and judgments retain their respective attribution and terms; see [provenance](docs/PROVENANCE.md).
