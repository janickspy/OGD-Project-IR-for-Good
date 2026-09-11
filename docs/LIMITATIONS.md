# Tool capabilities and limitations

| Area | Implemented behavior | Limitation |
|---|---|---|
| Recency | Validated normalized memberships, fixed reference date and visible date-status flags | Metadata age is not the age of the underlying observations |
| Ranking | Corpus-wide BM25, linear score, pure Mamdani and explicit hybrid | Policy choices are hand-specified rather than optimized |
| Explanations | One ranker produces scores, matched terms/fields, memberships, rules and exact blend traces | Technical faithfulness does not establish user comprehension |
| Sensitivity | Fourteen variations use the same evaluator and record the baseline hash | Local stability does not establish that a policy is appropriate |
| Candidate sets | Shared inherited judged pools with deterministic ties | This benchmark is not actual portal retrieval |
| BM25 | Actual corpus statistics and consistent field weights | No multilingual dense baseline is included |
| Judgment import | Invalid or blank grades are rejected before atomic replacement | Software cannot verify who assessed the labels |
| Grade scale | Published 0-2 grades are preserved with their declared source mapping | Original assessment lineage remains unverified |
| Empty positive pools | Separate all-query and positive-query aggregates | A pool with no positives is not a demonstrated corpus coverage gap |
| Resource counts | Counts and normalized support values are exposed | Download availability is not checked |
| Corpus | Audit of the supplied 500-record snapshot and publisher exposure | Records are concentrated among three publishers and one modification date |
| Runtime | Offline CLI and deterministic full-corpus scoring | Intended for inspection of small collections; production scaling is not evaluated |

The supplied results describe the new policy on inherited labels. They do not isolate the effect of individual changes from the predecessor. Broader conclusions require independent judgments and a more representative collection. See [evaluation workflow](EVALUATION.md) for preparing and validating additional inputs.
