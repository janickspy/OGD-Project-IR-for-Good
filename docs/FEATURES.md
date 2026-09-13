# Modules and capabilities

The Streamlit interface and command-line interface use the same application services. Python modules are located in `src/ogd_ir/`.

| Component | Module | Responsibility |
|---|---|---|
| Interface | `app.py` | Search, collection, analytics, assessment, evaluation, calibration and study workspaces |
| Command line | `cli.py` | Search, collection, portal capture, pooling, evaluation, calibration and analysis commands |
| Data model | `model.py` | Dataset validation, multilingual text normalization, tokenization and corpus hashes |
| Metadata catalog | `catalog.py` | CKAN label normalization, multilingual display fields, facets and resource links |
| Portal client | `portal.py` | Read-only API requests, retries, resumable snapshots and timestamped search-order capture |
| Query processing | `query.py` | Language and temporal-intent heuristics, vocabulary expansion and optional OpenAI assistance |
| Ranking | `ranking.py` | Corpus-wide BM25, feature extraction, Mamdani inference, linear and hybrid scores, explanation traces |
| Configuration | `config.py` | Policy validation, membership functions, weights and configuration hashes |
| Semantic retrieval | `semantic.py` | Pinned multilingual MiniLM inference on CPU, cosine ranking and checksummed embedding caches |
| Calibration | `calibration.py` | Distribution-based membership proposals with provenance and coverage validation |
| Analytics | `analytics.py` | Metadata audits, publisher exposure, rank movements and feature interventions |
| Annotation pools | `annotation.py` | Seeded candidate pooling and unassessed relevance records |
| Persistent records | `storage.py` | SQLite assessments, adjudication, agreement statistics, feedback and audit events |
| Evaluation | `evaluation.py` | Judgment validation, retrieval metrics, per-query reports, bootstrap intervals and paired comparisons |
| Study execution | `study.py` | Counterbalanced conditions, fixed protocols, response validation and timing |
| File handling | `io.py` | Strict JSON parsing, canonical hashes and atomic output replacement |

## Data flow

Collected metadata is normalized into a corpus and a display catalog. Ranking indexes the full selected corpus; filters restrict candidates without recomputing corpus statistics. The scoring call returns feature values, rule activations and scores together, so displayed explanations and evaluation use the same calculation.

Evaluation operates on explicit judgments. Unassessed grades remain null, and verified assessment exports require independent ratings and resolution of disagreements. Calibration produces a separate proposal; applying it changes the policy hash. Runtime records and downloaded model files are stored outside version control.

Analysis scripts are located in `scripts/`; automated scoring, integration and interface checks are in `tests/`.
