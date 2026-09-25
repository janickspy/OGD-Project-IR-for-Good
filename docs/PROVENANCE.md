# Attribution and provenance

The project is maintained by Janick Spycher and builds on Deep Shukla's master's thesis on human-centred retrieval of Swiss open-government data. Deep Shukla contributed the earlier research, application, metadata collection and relevance judgments.

The historical reanalysis uses two data artifacts from [Deep0901/Master_thesis_project](https://github.com/Deep0901/Master_thesis_project), pinned to commit `307d454248182cf4221a15d5853ae337da7c7798`. Source paths and SHA-256 hashes are recorded in `data/legacy/source_manifest.json`. `scripts/prepare_legacy.py` verifies these files and normalizes the data. Raw downloads are stored in `data/local/`, excluded from version control. Contact-point fields are omitted from the normalized corpus.

Metadata remains subject to its original publishers' terms; the software's MIT licence does not relicense it. The source judgment file describes a single-assessor consolidation and a 3-to-2 grade collapse (grade 3 merged into grade 2). Its published 0–2 grades are preserved exactly. According to the thesis author, the final grades were assigned by the author alone, in an annotation view that showed the query, the dataset title and identifier, the systems that had retrieved the dataset and existing grade fields, and no AI tool was used to assign or suggest grades. The raw grades behind the final labels are not preserved in the source repository, and its earlier heuristic labels (`evaluation/ground_truth_auto.json`) cover only four of the 150 final pairs, so the assessment history cannot be verified. The corresponding reports retain the status `inherited_unverified`.

## Metadata versions

`catalog.json` contains multilingual display labels, formats, dataset slugs and coverage from the checksum-verified metadata. `corpus_v2.json` normalizes historical and live records through the same human-label transformation. `normalization_v2.json` records the v1 and v2 corpus hashes and confirms the unchanged dataset ID set. The original `corpus.json` is retained for reproducing the v1 results.

## Semantic model

Semantic retrieval uses the Sentence Transformers model [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). Evaluation reports identify its immutable revision, ONNX artifact and SHA-256 hashes. Model weights are downloaded separately and retain the model publisher's licence and attribution.
