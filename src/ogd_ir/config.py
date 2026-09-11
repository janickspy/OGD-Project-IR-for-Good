"""Explicit, serializable policy. No calibration discovery or silent fallback."""
from copy import deepcopy
import math
from .io import read_json, digest

FEATURES = ("similarity", "completeness", "freshness", "resources")


def default_config():
    # Policy choices, not fitted to or optimized on the historical judgments.
    return {
        "schema_version": 1, "policy": "complete_rule_grid_v1",
        "reference_date": "2026-03-06T23:59:59Z",
        "bm25": {"k1": 1.5, "b": 0.75,
                 "field_weights": {"title": 3, "description": 1, "keywords": 2, "themes": 2}},
        "similarity_saturation": 1.0, "freshness_half_life_days": 730.0,
        "resource_cap": 10, "missing_freshness": 0.5,
        "alpha": 0.65, "feature_weights": {f: 1.0 for f in FEATURES},
        "memberships": {f: {"low": [0., 0., .5], "medium": [0., .5, 1.],
                             "high": [.5, 1., 1.]} for f in FEATURES},
        "output_memberships": {"very_low": [0., 0., .25], "low": [0., .25, .5],
                               "moderate": [.25, .5, .75], "good": [.5, .75, 1.],
                               "excellent": [.75, 1., 1.]},
        "rule_weights": {}, "universe_points": 1001,
    }


def triangle_valid(p):
    return (len(p) == 3 and all(isinstance(x, (float, int)) and math.isfinite(x) for x in p)
            and 0 <= p[0] <= p[1] <= p[2] <= 1 and p[0] < p[2])


def validate(config):
    expected = set(default_config())
    if set(config) != expected:
        raise ValueError(f"Configuration keys differ: {set(config) ^ expected}")
    from .model import timestamp
    timestamp(config["reference_date"])
    if config["schema_version"] != 1 or config["policy"] != "complete_rule_grid_v1":
        raise ValueError("Unsupported policy/schema")
    for key in ("alpha", "missing_freshness"):
        if not math.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f"Invalid {key}")
    for key in ("similarity_saturation", "freshness_half_life_days", "resource_cap"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if set(config["feature_weights"]) != set(FEATURES):
        raise ValueError("Feature weights must name all four features")
    if any(not math.isfinite(x) or x < 0 for x in config["feature_weights"].values()) or sum(config["feature_weights"].values()) <= 0:
        raise ValueError("Feature weights must be nonnegative with positive sum")
    bm = config["bm25"]
    if set(bm) != {"k1", "b", "field_weights"} or not math.isfinite(bm["k1"]) or bm["k1"] <= 0 or not 0 <= bm["b"] <= 1:
        raise ValueError("Invalid BM25 parameters")
    if set(bm["field_weights"]) != {"title", "description", "keywords", "themes"}:
        raise ValueError("Invalid BM25 fields")
    if any(type(x) is not int or x < 0 for x in bm["field_weights"].values()) or not sum(bm["field_weights"].values()):
        raise ValueError("BM25 repetition weights must be nonnegative integers")
    if type(config["universe_points"]) is not int or config["universe_points"] < 101:
        raise ValueError("Use at least 101 universe points")
    if set(config["memberships"]) != set(FEATURES):
        raise ValueError("Missing input membership family")
    families = list(config["memberships"].values()) + [config["output_memberships"]]
    for family in families:
        if any(not triangle_valid(p) for p in family.values()):
            raise ValueError("Degenerate, nonfinite, unordered or out-of-domain triangle")
        # Require endpoint shoulders and overlapping support; reject coverage gaps.
        supports = sorted(family.values())
        if not any(p[0] == p[1] == 0 for p in supports) or not any(p[1] == p[2] == 1 for p in supports):
            raise ValueError("Membership families need left and right shoulders")
        end = supports[0][2]
        for p in supports[1:]:
            if p[0] >= end:
                raise ValueError("Membership support has a gap or uncovered junction")
            end = max(end, p[2])
    if any(set(f) != {"low", "medium", "high"} for f in config["memberships"].values()):
        raise ValueError("Input term vocabulary mismatch")
    if set(config["output_memberships"]) != {"very_low", "low", "moderate", "good", "excellent"}:
        raise ValueError("Output term vocabulary mismatch")
    if any(k not in {f"R{i:03}" for i in range(81)} or not math.isfinite(v) or v <= 0
           for k, v in config["rule_weights"].items()):
        raise ValueError("Unknown rule or nonpositive rule weight")
    return config


def load_config(path=None):
    return validate(read_json(path) if path else deepcopy(default_config()))


def config_hash(config):
    return digest(validate(config))
