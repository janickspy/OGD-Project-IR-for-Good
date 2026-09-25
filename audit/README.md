# Ranking audit

Scripts and results for the audit of the transparent multi-criteria ranker on the normalised corpus (`data/legacy/corpus_v2.json`) with the default configuration (`configs/default.json`).

| Check | Scripts | Result files |
|---|---|---|
| C1 Trace fidelity | `analyze.py`, `trace_consistency.py` | `results/trace_validation.json`, `results/trace_consistency.json`, `results/pooled_traces.json` |
| C2 Feature variation | `analyze.py` | `results/feature_diagnostics.json` |
| C3 Constant-value interventions | `analyze.py` | `results/feature_interventions.json` |
| C4 Monotonicity and local variants | `analyze.py`, `resource_counterfactuals.py`, `policy_checks.py` | `results/monotonicity_grid.json`, `results/observed_feature_counterfactuals.json`, `results/resource_counterfactuals.json`, `results/policy_checks.json` |
| C5 Publisher exposure | `policy_checks.py` | `results/policy_checks.json` |
| C6 Judgment coverage and pool extension | `build_judgment_pool.py`, `german_bm25.py`, `analyse_judgments.py`, `spot_check.py` | `results/full_corpus_runs.json`, `results/german_bm25_runs.json`, `results/full_corpus_candidate_coverage.json`, `results/judgment_analysis.json` |

The pooled reranking figures are in `../results/application_reanalysis.json` (produced by `scripts/run_semantic_research.py`). The paired tests reported with them are the exact permutation tests with Holm correction in `results/exact_paired_tests.json`; the Wilcoxon approximations stored in the application report are not used.

## Reproduction

After installing the package (`python -m pip install -e '.[semantic,audit]'`), from the repository root:

```sh
python audit/analyze.py
python audit/trace_consistency.py --model-cache data/local/models
python audit/build_judgment_pool.py --model-cache data/local/models
python audit/german_bm25.py
python audit/build_judgment_pool.py --round2
python audit/policy_checks.py
python audit/resource_counterfactuals.py
python audit/analyse_judgments.py
python audit/paper_numbers.py
```

`paper_numbers.py` prints every number reported in the paper from the result files. The `audit` extra installs the libraries used by `german_bm25.py`, snowballstemmer (BSD-3-Clause) and compound-split (GPL-3.0, which packages the CharSplit model), and openpyxl for the spot-check sheet.

The semantic model is downloaded once with `python scripts/run_semantic_research.py`, which also rebuilds the application report. Without `--model-cache`, `build_judgment_pool.py` reuses `results/full_corpus_runs.json` and `trace_consistency.py` skips the semantic check. `resource_counterfactuals.py` also renders the counterfactual figure if matplotlib is installed.

`analyze.py` checks that corpus, configuration and inherited judgments match the digests of the application report. An anonymised review copy of this repository replaces personal names in the provenance fields of `data/legacy/judgments.json`, which changes the judgment digest. In that case, run it with `OGD_ALLOW_PROVENANCE_EDIT=1`; it then verifies the grades themselves against a stored digest.

## Inherited judgments

The 150 inherited grades (`data/legacy/judgments.json`) come from the earlier thesis project, in which the thesis author, as the only assessor, graded a fixed set of ten candidates per topic; all systems rerank these candidates. According to the thesis author, the annotation view showed the query, each candidate's title and identifier, the systems that had retrieved it and existing grade fields. There is no record that descriptions, keywords or portal pages were consulted, and no AI tool was used to assign or suggest grades. The thesis describes heuristic pre-labelling and a consolidation pass, but the heuristic labels preserved in the thesis repository (`evaluation/ground_truth_auto.json`, 450 labels) cover only four of the 150 final pairs, and the raw grades behind the final labels were not kept. The grading therefore cannot be reconstructed. The grades are used as published and have not been re-assessed; their status is `inherited_unverified`.

## Extended judgments

The full-collection evaluation pools the top ten results of all five systems for each topic together with the inherited candidates. This gives 451 query–dataset pairs, 301 of them without inherited labels. Two large language model assessors graded all 451 pairs:

- Assessor A: Claude Opus 5.5 (Anthropic)
- Assessor B: Claude Sonnet 5 (Anthropic)

Both received the same guideline (`judgments/guideline.md`) and topic definitions (`judgments/topics.json`, query text and the benchmark's one-sentence information need). Their inputs were the batch files in `judgments/`: topics grouped by domain, items shuffled within each topic with a different seed per assessor, and opaque item ids (`judgments/pool_key.json` maps them to query and dataset ids). Each record shows the title and description in all available languages (description cut at 1,500 characters), at most 20 keywords, English labels of the theme codes, the publisher name in all available languages and, where available, spatial coverage, update frequency and formats. Modification dates, resource counts, links, system names, ranks, scores and inherited labels were not shown.

Each batch was judged in a separate session in which the assessor was instructed to read only the guideline and its batch file and to use no other files or web sources. The assessors ran as tool-using agents without explicitly set sampling parameters. Their outputs are kept unchanged in `judgments/raw/`; `judgments/assessor_A.json` and `judgments/assessor_B.json` join them with the pool key and sort them, without changing grades or reasons.

`build_judgment_pool.py` regenerates the batch files deterministically. The files in `judgments/` are byte-identical to those given to the assessors:

| File | SHA-256 |
|---|---|
| `batch_A_ENV.md` | `5ec8305b4453d13ce5361b1f93079caeb04ad0237418754592d00c72c6f569d1` |
| `batch_A_MOB.md` | `727b7eb052df41344a67582b69776d8da8d410a07d3af0586ed1d756c4014d51` |
| `batch_A_XD.md` | `df42713f42b50e92e1a3f547cf91cc07157b89d44c89e8fb1fc856d75a3a5460` |
| `batch_B_ENV.md` | `66060bd825f7da27dffff2c263c43560e6fbeb66b09824606a424380ba7fafe2` |
| `batch_B_MOB.md` | `4de3c60c5172bc25950a455c04072b66c43a39a35f99f7525353f91493f235ae` |
| `batch_B_XD.md` | `b5f08c73480256fa1bfd3375b22da20c71322af73b4f11541e0ddb98c7d6d58b` |
| `guideline.md` | `9958e562911ec92bb5a96fdd2708466cab70985250f7715c3a6801d387b8c488` |

### Second round

After the first analysis, two BM25 variants were added (`german_bm25.py`): `bm25_stem` removes the Snowball German stopwords (`german_stopwords.txt`) and applies the Snowball German stemmer, and `bm25_split` also indexes the CharSplit parts of words with at least eight letters when the split score is at least 0.5, splitting each part once more and keeping the original word. Both use the application's fields, field weights, k1 and b and were not tuned. `build_judgment_pool.py --round2` pools their full-collection top ten results that were not in the first-round pool. This gives 141 pairs, none with inherited labels. Both assessors graded them with the same guideline, record rendering and procedure: three domain batches judged in separate sessions, items shuffled within each topic with new seeds (3307 for A, 4409 for B). Inputs, unchanged outputs and joined files are in `judgments/round2/`, with the same structure as the first round. The two rounds cover 592 pairs, 442 of them without inherited labels, and include the complete top ten results of all seven systems. The round-2 batch files are byte-identical to those given to the assessors:

| File | SHA-256 |
|---|---|
| `round2/batch_A_ENV.md` | `8d16d63c8450530a7b9772a2d870d78ae8e9bb05bbf59e537efbbcf08f51ccc9` |
| `round2/batch_A_MOB.md` | `687601f819ba97b3e70c1dcbc6292d2999f08c3398da4ba1865613ccb47a5c38` |
| `round2/batch_A_XD.md` | `f978d5adf7594ce1be71dd7dfa41be2b1809fdb1507fca33d109f4082397ea7e` |
| `round2/batch_B_ENV.md` | `58bfaaf9ad7530faa08bd7046392d181ca942d822ae73560185ae8fe04bc09f0` |
| `round2/batch_B_MOB.md` | `907bc100303c73b7b065ce94c68b3e4f09a2c3f614843c56cad56371aaff9fd1` |
| `round2/batch_B_XD.md` | `d0a2c76462e7e30d3abf67ce30779032b09d0569aada20978540b09376c63819` |

`analyse_judgments.py` merges both rounds. Pairwise tests among the five audited systems form the primary Holm family; the comparisons involving the German variants, which were added later, form a second family.

### Use of the labels

These labels are model judgments, not human assessments. `results/judgment_analysis.json` reports their agreement with the inherited labels and with each other, and evaluates every system separately under each assessor, with only grade 2 counted as relevant, and with statistical yearbooks counted as not relevant.

`spot_check.py sample` draws 40 pairs without and 20 pairs with inherited labels from the first-round pool into a blind grading sheet (`judgments/spot_check_sheet.xlsx`); `spot_check.py score <sheet> [<sheet> ...]` compares each completed sheet with both assessors and with the inherited labels and, when several people grade the sheet independently, the sheets with each other.
