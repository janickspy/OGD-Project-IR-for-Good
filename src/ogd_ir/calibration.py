"""Explicit, versioned calibration proposals. Never change an active policy silently."""
from copy import deepcopy
import numpy as np
from .config import FEATURES, validate, config_hash
from .io import digest


def propose_calibration(ranker, queries):
    if not queries or any(not q.strip() for q in queries):raise ValueError('Calibration needs nonempty queries')
    values={f:[] for f in FEATURES}
    for d in ranker.corpus.datasets:
        metadata,_=ranker.features(queries[0],d)
        for f in FEATURES[1:]:values[f].append(metadata[f])
        for query in queries:
            features,evidence=ranker.features(query,d)
            if evidence['bm25']>0:values['similarity'].append(features['similarity'])
    config=deepcopy(ranker.config);decisions={}
    for feature, samples in values.items():
        distinct=len(set(samples))
        if len(samples)<5 or distinct<3:
            decisions[feature]={'changed':False,'reason':'Insufficient variation; existing memberships retained',
                                'n':len(samples),'distinct':distinct};continue
        median=float(np.median(samples));peak=min(.9,max(.1,median))
        family={'low':[0,0,peak],'medium':[0,peak,1],'high':[peak,1,1]}
        decisions[feature]={'changed':family!=config['memberships'][feature],'n':len(samples),'distinct':distinct,
                            'median':median,'clipped_peak':peak}
        config['memberships'][feature]=family
    validate(config)
    return {'config':config,'config_hash':config_hash(config),'parent_config_hash':ranker.policy_hash,
            'corpus_hash':ranker.corpus.hash,'query_hash':digest(queries),'queries':queries,
            'method':'Median normalized feature peak clipped to [0.1,0.9], full-domain shoulders; no relevance fitting',
            'decisions':decisions,'reference_date':config['reference_date'],
            'warning':'A proposal, not an automatic improvement. Evaluate on separately judged queries before adoption.'}
