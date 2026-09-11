# Executable method

The reference is 2026-03-06 23:59:59 UTC. Naive source timestamps are interpreted as UTC. Invalid, missing or future dates receive freshness 0.5 and a visible status flag, and fail the completeness date check. Valid freshness is 2^(-age_days/730). This is metadata recency, not the temporal coverage of the underlying data.

BM25 indexes all normalized documents with actual N and document frequencies. Unicode NFKC/casefold word tokens are used without stemming, stop-word removal or query expansion. Dictionary-valued multilingual fields concatenate in sorted key order. Repetition weights: title 3, description 1, keywords 2, themes 2. k1=1.5, b=0.75. Query terms are unique. IDF=log(1+(N-df+0.5)/(df+0.5)). Similarity is raw/(raw+1) times the proportion of query terms found. These choices can be poor for morphological variants and cross-language search.

Completeness is the mean of eight checks: nonempty title, description >=50 characters, publisher, keywords, themes, license, positive resource count, and valid nonfuture modification date. Resources=min(log(1+count)/log(11),1). Both are proxies; neither verifies data correctness or download success.

All four inputs use low=(0,0,.5), medium=(0,.5,1), high=(.5,1,1). Output terms very_low, low, moderate, good, excellent use triangles (0,0,.25), (0,.25,.5), (.25,.5,.75), (.5,.75,1), (.75,1,1). Shoulder endpoints are handled explicitly; singleton triangles are rejected.

The 81 rules enumerate similarity S, completeness C, freshness T and resources A in {0,1,2}. Consequent index is 0 when S=0, otherwise S+min(C,max(T,A)). This ordinal policy limits metadata support by completeness and allows freshness or resource count to supply that support. It is a new hand-specified policy, not the predecessor's rule base. Rules combine antecedents by minimum; firing=min(1,weight*minimum), default weight 1. Consequents are clipped and aggregated by maximum. Discrete centroid uses 1001 equally spaced points on [0,1], including endpoints. It is a specified numerical approximation, not an exact continuous integral.

Linear score is the weighted mean of four features (equal weights by default). Hybrid=.65*centroid+.35*linear. All systems finally require raw BM25>0, else score=0. Deterministic ties sort by dataset ID. Linear and Mamdani are separately evaluated comparators: the hybrid is never labelled pure fuzzy inference.

The current 500-document implementation favors clarity over scalability: search evaluates every document, and traces include all activated rules. It is an offline research engine, not a production portal deployment. No assertion of superiority, fairness, human comprehension or global monotonicity follows from inspection alone.
