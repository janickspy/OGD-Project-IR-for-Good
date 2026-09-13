# Capabilities and limits

The bundled example supports offline lexical retrieval and evaluation. Semantic retrieval requires a model download on first use. Live collection and optional OpenAI query assistance depend on external services.

## Historical evidence

The included dataset snapshot contains 500 records, heavily concentrated in a small number of publishers. The 15 queries have 150 inherited 0–2 judgments, with six pools containing no positive labels. Those six pools do not establish that the full collection lacks relevant data. Assessment provenance remains `inherited_unverified`; no new independent assessor or participant study is claimed.

The four-method historical reanalysis, five-method semantic extension and application-normalization reanalysis are distinct runs. The v2 normalization removes internal tag identifiers and state fields from indexed text and retains portal URLs; the original v1 corpus and inherited grades remain available. Differences from the earlier application involve multiple design changes and are not controlled estimates of effectiveness improvement.

## Interpretation

Completeness is metadata presence, freshness is metadata modification age, and resource count is a count rather than verified availability. Policy priorities can disadvantage historical datasets or publishers with sparse metadata. Exposure diagnostics show consequences in the selected candidates; they do not certify fairness or social benefit. Membership stability is not evidence of explanation usefulness.

The handcrafted 81-rule policy differs from the predecessor's implementation. Scalar feature weights apply to the linear component and its contribution to the hybrid. The Mamdani rule policy is configured separately. There is no claim that fuzzy reasoning outperforms the simpler comparator.

Semantic search uses a pinned quantized ONNX export rather than reproducing the earlier application's model execution. Truncation, multilingual text order, runtime and quantization can affect results. Semantic similarity is not a field-level causal explanation.

## Operational limits

CKAN collection cannot freeze the changing portal transactionally. Capture timestamps and dataset coverage must be reconciled before comparison with frozen snapshots. Checkpoints preserve partial progress; they do not guarantee complete coverage if the portal changes during collection.

The application is local and has no account authentication or multi-tenant authorization. Assessor independence and participant consent require a facilitator. Participant mode hides administrative controls but is not a public-service security boundary. Local assessment data is excluded from version control and requires separate backup.

The study runner supports response collection, counterbalancing and export. These capabilities do not establish a validated design, adequate sample size or improved comprehension. Default tasks require review and piloting; the bundled results contain no participant observations.
