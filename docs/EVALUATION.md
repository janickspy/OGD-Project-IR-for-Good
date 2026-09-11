# Evaluation and annotation workflow

## Reproduce the supplied benchmark

Run `make research` from the repository root. This verifies frozen source checksums, normalizes the corpus and judgments, and regenerates the benchmark, sensitivity results, corpus audit, publisher exposure and explanation example. Evaluation uses `configs/default.json`; configuration, corpus and judgment hashes are recorded in the outputs.

The supplied labels have status `inherited_unverified`, so direct evaluation requires an explicit opt-in:

```sh
ogd-ir --corpus data/legacy/corpus.json evaluate --judgments data/legacy/judgments.json --allow-unverified --output results/legacy_reanalysis.json
```

All four systems reorder the same ten-record pools. AP@10 uses all known positive judgments as its denominator; binary relevance is grade greater than zero. P@5 has denominator five, nDCG@10 uses gain 2^grade-1, and reciprocal rank is truncated at ten. Queries without known positive judgments score zero and are also separated from the positive-query aggregate. Reports include query bootstrap intervals and one six-comparison Holm family for nDCG@10.

## Prepare new judgments

Generate a randomized pool from the tool's four ranking systems:

```sh
PYTHONPATH=src python3 scripts/create_annotation_pool.py
```

This writes `data/local/annotation_pool.json` with `null` grades. Null means unassessed and is deliberately rejected by the evaluator. The generator pools up to 20 results per system and records the policy, corpus, systems, depth and random seed.

Assign integer grades 0=irrelevant, 1=partly relevant and 2=highly relevant. Retain assessor identifiers, instructions, timestamps and rationales in separate source records. For independent reassessment, preserve each assessor's original labels and any adjudication separately. Do not silently convert blank labels to zero or collapse other grading scales. Mark the consolidated data `verified` only after its provenance has been checked; that field is an attestation by the data provider, not authentication by the program.

Import the completed file with validation before replacement:

```sh
ogd-ir --corpus data/legacy/corpus.json import-judgments --input assessed.json --output data/local/verified.json
```

The importer rejects unknown dataset IDs, duplicate JSON keys and missing or invalid grades before modifying the destination file. Use `--mode full_corpus` to evaluate all-corpus retrieval only when the returned results have judgments; unjudged returned records raise an error. Even fully judged returned results do not establish exhaustive recall.

## Extend the collection

Freeze and document the collection date separately from metadata modification and observation coverage. Include a wider range of publishers, domains and languages. Preserve dataset URLs, original licenses and missing-field information. Use separate development and evaluation queries when adjusting the policy. Additional baselines need captured outputs, versioned configurations and an expanded judgment pool; a lexical heuristic must not be presented as an actual portal or semantic run.
