# Evaluation and annotation workflow

## Reproduce the supplied benchmark

`make research` verifies frozen source checksums, normalizes the corpus and judgments, and regenerates the benchmark, sensitivity results, corpus audit, publisher exposure and explanation example. Evaluation uses `configs/default.json`; configuration, corpus and judgment hashes are recorded in the outputs.

The supplied labels have status `inherited_unverified`, so direct evaluation requires an explicit opt-in:

```sh
ogd-ir --corpus data/legacy/corpus.json evaluate --judgments data/legacy/judgments.json --allow-unverified --output results/legacy_reanalysis.json
```

All four systems reorder the same ten-record pools. AP@10 uses all known positive judgments as its denominator; binary relevance is grade greater than zero. P@5 has denominator five, nDCG@10 uses gain 2^grade-1, and reciprocal rank is truncated at ten. Queries without known positive judgments score zero and are also separated from the positive-query aggregate. Reports include query bootstrap intervals and one six-comparison Holm family for nDCG@10.

## Relevance judgments

Randomized pooling across the four lexical ranking methods:

```sh
PYTHONPATH=src python3 scripts/create_annotation_pool.py
```

The output is `data/local/annotation_pool.json` with `null` grades. Null denotes an unassessed item and is rejected by the evaluator. The generator pools up to 20 results per system and records the policy, corpus, systems, depth and random seed.

The relevance scale is 0=irrelevant, 1=partly relevant and 2=highly relevant. Assessment provenance comprises assessor identifiers, assessment instructions, timestamps, rationales and any adjudication records. Missing labels are not treated as zero, and conversion from other grading scales requires a documented mapping. The `verified` status records the data provider's attestation following provenance review; it is not identity authentication.

Validated import:

```sh
ogd-ir --corpus data/legacy/corpus.json import-judgments --input assessed.json --output data/local/verified.json
```

The importer rejects unknown dataset IDs, duplicate JSON keys and missing or invalid grades before modifying the destination file. The `--mode full_corpus` option evaluates retrieval across the selected corpus and requires judgments for returned records. Unjudged returned records raise an error. Fully judged returned results do not establish exhaustive recall.

## Collection and experimental scope

Collection time, metadata modification time and observation coverage describe different properties. Snapshot provenance records collection dates, publisher and language coverage, dataset URLs, licences and missing fields. Policy development and evaluation require separate query sets to avoid fitting to test judgments. Additional baselines require captured outputs, versioned configurations and judgments covering their results.

## Application workflows

The interface builds persistent blind pools and supports independent assessments and adjudication. Candidate order is stored explicitly to preserve randomized presentation through serialization. Agreement is computed over shared rated pairs; weighted kappa is undefined when expected disagreement is zero. Verified export requires at least two grades per item, resolution of disagreement and facilitator attestation. A changed grade invalidates any earlier adjudication.

Semantic retrieval adds a fifth system to the common judged pool. All ten pairwise nDCG comparisons then form one Holm family; the four-system family has six comparisons. Reports record the model and corpus hashes. Full-corpus evaluation rejects unjudged returned items. Pool verification does not establish exhaustive recall.

`results/application_reanalysis.json` uses the application v2 human-label normalization. `results/semantic_reanalysis.json` extends the historical v1 input run. Configurations, corpus hashes and judgment hashes are embedded in each report. Live portal orders are captured separately and are not included as a baseline in these frozen comparisons.
