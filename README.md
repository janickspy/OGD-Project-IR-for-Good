# OGD dataset workbench

A Python application for Swiss open-government dataset search and evaluation. It combines lexical and semantic retrieval with interpretable multi-criteria ranking, metadata analysis and relevance assessment.

Maintained by Janick Spycher. Research and data sources are documented in [provenance](docs/PROVENANCE.md).

## Installation

Python 3.11 or 3.12 is required.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[app,semantic,dev]'
python -m streamlit run app.py
```

The interface runs at `http://localhost:8501` by default. The bundled 500-dataset example supports offline lexical search. Semantic retrieval downloads a pinned multilingual MiniLM model on first use and caches it locally. The `app,dev` installation extras omit semantic-model dependencies.

The server binds to localhost and does not provide account authentication. Runtime data is stored in `data/local/`, excluded from version control. `OGD_WORKSPACE` overrides this location.

## Workspaces

| Workspace | Available functionality |
|---|---|
| Search | DE/FR/IT/EN metadata, query interpretation, optional multilingual expansion, publisher/theme/format/licence/language filters, pagination, portal/resource links and result exports |
| Ranking comparison | BM25, weighted linear ranking, Mamdani inference, hybrid ranking and multilingual semantic retrieval; feature plots and scoring traces |
| Collection | Read-only CKAN search, paged collection, retries, resumable checkpoints, normalized snapshots, manifests and actual portal-order capture |
| Analytics | Publisher representation, metadata omissions and dates, query-specific rank changes, exposure summaries, feature interventions and opt-in local feedback |
| Assessments | Blind candidate pools, independent 0/1/2 grades, saved notes, overlap agreement, quadratic weighted kappa, adjudication and guarded export |
| Evaluation | Shared candidate pools, strict judgments, per-query results, all-query/positive-query aggregates, bootstrap intervals and Holm-adjusted paired comparisons |
| Calibration | Membership plots, explicit configuration import/export, distribution-based calibration proposals and deliberate adoption |
| Study | Fixed protocols, three counterbalanced conditions, explanations shown/hidden, simpler linear comparator, objective comprehension responses, clarity ratings and elapsed time |

Ranking criteria represent metadata properties and policy preferences. Their interpretation and limits are described in [method](docs/METHOD.md) and [limitations](docs/LIMITATIONS.md).

## Tests and evaluation

```sh
python -m pytest -q
python scripts/run_research.py
python scripts/run_semantic_research.py
```

`run_research.py` evaluates four lexical ranking methods on the historical v1 corpus. `run_semantic_research.py` adds semantic retrieval and produces reports for [v1](results/semantic_reanalysis.json) and [normalized metadata labels](results/application_reanalysis.json). The latter indexes tag labels rather than internal UUID and state fields. Both versions contain the same 500 dataset IDs, 15 query texts and 150 relevance grades.

The supplied results concern reranking of ten-document pools with inherited, unverified relevance labels. They do not measure end-to-end portal retrieval. Reports include model, corpus and configuration provenance, per-query metrics and uncertainty estimates.

`make research` also downloads the checksum-verified source files and rebuilds the normalized inputs. The included inputs can be evaluated without downloading the sources again.

## Command line

Global options precede the subcommand:

```sh
ogd-ir --corpus data/legacy/corpus_v2.json search 'historical population 1990' --expand --explain
ogd-ir --corpus data/legacy/corpus_v2.json search 'Swiss population statistics' --system semantic
ogd-ir collect --output data/local/snapshots/swiss-example --limit 500
ogd-ir collect --output data/local/snapshots/swiss-example --limit 500 --resume
ogd-ir --corpus data/legacy/corpus_v2.json analyse --query 'Verkehr' --output data/local/analysis.json
ogd-ir --corpus data/legacy/corpus_v2.json evaluate --judgments data/legacy/judgments.json --allow-unverified --semantic --output data/local/evaluation.json
```

For `pool`, `capture-portal` and `calibrate`, `--queries` accepts a JSON list of objects with `id` and `text`, or an object containing that list under `queries`. Calibration outputs contain a proposed configuration and its provenance. The interface also supports configuration import and export.

Optional OpenAI query assistance requires `OPENAI_API_KEY` and a model name. A request is sent only when **Suggest a query** is selected; suggestions are not applied automatically. Other features do not require this API key.

## Documentation

- [Application operation](docs/APPLICATION.md)
- [Modules and capabilities](docs/FEATURES.md)
- [Scoring method](docs/METHOD.md)
- [Evaluation and assessment](docs/EVALUATION.md)
- [Capabilities and limits](docs/LIMITATIONS.md)
- [Attribution and data provenance](docs/PROVENANCE.md)

## Licence

The software is released under the MIT licence. Metadata, relevance judgments and model weights retain their respective attribution and terms.
