# Operating the application

## Data and configuration

The sidebar selects a snapshot and display language. The default example uses `data/legacy/corpus_v2.json`; the historical v1 snapshot preserves the earlier ranking inputs. The accompanying `catalog.json` contains multilingual labels, dataset slugs, formats, coverage and other display metadata. Live snapshots have their own corpus, catalog, checkpoint and manifest. Resource URLs are absent from the historical source; these records link to their dataset pages on the portal.

BM25 statistics always come from the complete selected snapshot, including when filters restrict displayed candidates. Four lexical methods share the same candidate set and term gate. Semantic retrieval uses the complete selected snapshot and its own cosine scores; it can return datasets without literal query-term matches. Filters are applied to all methods. A score from one method is not directly comparable to a score from another.

The reference date is fixed and visible. A newly collected snapshot initially uses its collection completion time. The historical example uses 6 March 2026. Custom dates or configuration changes generate a different configuration hash. Feature weights apply to the linear component; fuzzy rule weights are configured separately. Calibration produces a proposal that requires explicit adoption.

## Live collection and portal capture

The CKAN client uses HTTPS, timeouts and bounded retries for connection failures, timeouts, 429 and server errors. Other failures, including 403, are reported. No credentials are required for the public API. Page checkpoints are written atomically. Resume requires identical query, filters, page size and limit. The original local raw records are retained in the checkpoint; exported normalized metadata omits contact fields.

Collection uses stable ID ordering and deduplicates records. This reduces pagination problems but does not make the live portal transactional. The manifest records the observed counts, time range, limit and hashes. A limit of 500 is a sample, not the complete portal. Each snapshot occupies a separate directory.

Portal-order capture records returned dataset IDs, query, time, endpoint and sort order. Comparison with a frozen-corpus run requires compatible dataset coverage and assessment scope. A dataset absent from a capture has no observed portal rank.

## Semantic search

The model is `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, using `onnx/model_quint8_avx2.onnx`. ONNX Runtime executes it locally on CPU. Tokenization truncates to 128 tokens; attention-mask mean pooling is followed by L2 normalization. Scores are cosine similarities. The document text concatenates title, description, keywords and themes; long multilingual metadata may be truncated. Quantization can produce differences from floating-point PyTorch inference.

The cache key includes corpus hash, text policy and encoder provenance, including artifact and tokenizer SHA-256, runtime version and batch sizes. Cached arrays have a byte checksum and are loaded with pickle disabled. A failed or unavailable model raises an error. First use requires model download; the CLI `--offline-model` option requires the files to exist already.

Model documentation: [publisher model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). CKAN usage: [opendata.swiss API handbook](https://handbook.opendata.swiss/content/nutzen/api-nutzen.html).

## Local assessment and feedback

`data/local/workbench.sqlite` stores pseudonymous assessment records, adjudications, append-only change events, feedback and study trials. Assessor independence is established by the study procedure rather than application authentication. Other assessors' grades are hidden in the independent-assessment view. Agreement and adjudication are handled in a separate facilitator view. Changing an assessment invalidates the corresponding adjudication.

Feedback collection is opt-in. Submitting a usefulness rating records the query, dataset, ranking method and hashes. The Analytics workspace provides exports. Assessment and participant data are excluded from version control by default.

## Controlled explanation study

The Study workspace stores a fixed protocol with three conditions: hybrid ranking without explanation controls, hybrid ranking with explanations, and linear ranking with explanations. Each query appears once per participant. Sequential participant slots rotate the conditions, and query order is seeded within groups of three slots. A protocol is bound to its snapshot and policy hashes; changing either invalidates it for the current selection.

The default questions concern metadata recency, resource-count limitations and fuzzy inference. They require piloting for comprehension and learning effects across repeated questions. The linear condition changes both system and explanations, so it cannot isolate a causal fuzzy-rule effect. The hybrid shown/hidden comparison holds the ranking method fixed.

Participant-mode configuration:

```sh
OGD_STUDY_MODE=participant python -m streamlit run app.py
```

Participant mode hides other workspaces, facilitator setup and exports. It is intended for facilitated local sessions and does not provide public-server access control. Consent procedures are part of the study protocol; the interface additionally requires participant opt-in before recording responses. Timing begins when a task is started. Completed responses are immutable and persist across page reloads. The repository contains no participant observations or study outcomes.

## Validation

`python -m pytest -q` exercises scoring invariants, data preservation, live-client failure/pagination boundaries, strict imports, independent assessments, adjudication invalidation, semantic candidate alignment, study counterbalancing and Streamlit interface workflows. Network tests use explicitly named test doubles; actual semantic-model evaluation is reproduced separately with `python scripts/run_semantic_research.py`. Interface tests use Streamlit's AppTest and do not substitute for a graphical cross-browser inspection.
