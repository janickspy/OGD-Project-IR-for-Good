"""Canonical BM25, feature extraction, Mamdani inference and faithful traces."""
from collections import Counter
from copy import deepcopy
from itertools import product
import math
import numpy as np
from .config import FEATURES, load_config, validate, config_hash
from .model import tokens, timestamp


def triangle(x, points):
    a, b, c = points
    if not a <= b <= c or a == c:
        raise ValueError("A triangle requires ordered, nondegenerate support")
    values = np.asarray(x, dtype=float)
    left = np.ones_like(values) if a == b else (values - a) / (b - a)
    right = np.ones_like(values) if b == c else (c - values) / (c - b)
    return np.clip(np.minimum(left, right), 0., 1.)


def rules(config):
    """81 exhaustive linguistic rules, with an explicit ordinal policy.

    Low similarity -> very low. Otherwise metadata support is limited by
    completeness and can be supplied by freshness OR resource count.
    This is an inspectable policy choice, not an empirically learned rule base.
    """
    terms = ["low", "medium", "high"]
    outputs = ["very_low", "low", "moderate", "good", "excellent"]
    result = []
    for i, (s, c, t, a) in enumerate(product(range(3), repeat=4)):
        quality = min(c, max(t, a))
        consequent = outputs[0 if s == 0 else s + quality]
        rid = f"R{i:03}"
        result.append({"id": rid, "if": dict(zip(FEATURES, [terms[x] for x in (s, c, t, a)])),
                       "then": consequent, "weight": config["rule_weights"].get(rid, 1.0)})
    return result


class BM25:
    def __init__(self, corpus, config):
        self.config = config
        self.corpus = corpus
        self.docs = {}
        self.df = Counter()
        self.fields = {}
        for d in corpus.datasets:
            fields = {f: Counter(tokens(getattr(d, f))) for f in config["field_weights"]}
            self.fields[d.id] = fields
            counts = Counter()
            for field, weight in config["field_weights"].items():
                for term, n in fields[field].items():
                    counts[term] += n * weight
            counts = +counts
            self.docs[d.id] = counts
            self.df.update(counts.keys())
        self.lengths = {key: sum(v.values()) for key, v in self.docs.items()}
        self.avgdl = sum(self.lengths.values()) / len(corpus.datasets) or 1.0

    def score(self, query, identity):
        query_terms = sorted(set(tokens(query)))
        counts = self.docs[identity]
        score = 0.
        contributions = []
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            n, df = len(self.docs), self.df[term]
            idf = math.log1p((n - df + .5) / (df + .5))
            k1, b = self.config["k1"], self.config["b"]
            part = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.lengths[identity] / self.avgdl))
            score += part
            contributions.append({"term": term, "bm25_contribution": part,
                                  "fields": {f: counts_[term] for f, counts_ in self.fields[identity].items()
                                             if counts_.get(term) and self.config["field_weights"][f] > 0}})
        return score, contributions, len(query_terms)


class Ranker:
    def __init__(self, corpus, config=None):
        self.corpus = corpus
        self.config = validate(deepcopy(config)) if config is not None else load_config()
        self.policy_hash = config_hash(self.config)
        self.reference = timestamp(self.config["reference_date"])
        self.index = BM25(corpus, self.config["bm25"])
        self.rules = rules(self.config)
        self.universe = np.linspace(0., 1., self.config["universe_points"])
        self.output_curves = {term: triangle(self.universe, points)
                              for term, points in self.config["output_memberships"].items()}

    def features(self, query, d):
        raw, matches, query_count = self.index.score(query, d.id)
        coverage = len(matches) / query_count if query_count else 0.
        similarity = raw / (raw + self.config["similarity_saturation"]) * coverage
        # Presence proxy with fixed, published binary checks. Not intrinsic data quality.
        checks = {"title": bool(d.title.strip()), "description": len(d.description.strip()) >= 50,
                  "publisher": bool(d.publisher.strip()), "keywords": bool(d.keywords.strip()),
                  "themes": bool(d.themes.strip()), "license": bool(d.license.strip()),
                  "resources": d.resource_count > 0, "modified": False}
        age, status = None, "missing"
        if d.modified:
            try:
                age = (self.reference - timestamp(d.modified)).total_seconds() / 86400.
                status = "valid" if age >= 0 else "future"
            except (ValueError, TypeError, OverflowError):
                status = "invalid"
        checks["modified"] = status == "valid"
        freshness = (2 ** (-age / self.config["freshness_half_life_days"]) if status == "valid"
                     else self.config["missing_freshness"])
        values = {"similarity": similarity, "completeness": sum(checks.values()) / len(checks),
                  "freshness": freshness,
                  "resources": min(math.log1p(d.resource_count) / math.log1p(self.config["resource_cap"]), 1.)}
        details = {"bm25": raw, "query_coverage": coverage, "matched_terms": matches,
                   "completeness_checks": checks, "metadata_modified": d.modified,
                   "reference_date": self.config["reference_date"], "age_days": age,
                   "date_status": status, "resource_count": d.resource_count}
        return values, details

    def infer(self, values):
        if set(values) != set(FEATURES) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
            raise ValueError("Inference needs all four finite normalized features")
        memberships = {f: {t: float(triangle(v, p)) for t, p in self.config["memberships"][f].items()}
                       for f, v in values.items()}
        strengths = {term: 0. for term in self.output_curves}
        active = []
        for rule in self.rules:
            antecedents = {f: memberships[f][t] for f, t in rule["if"].items()}
            firing = min(1., min(antecedents.values()) * rule["weight"])
            if firing > 0:
                active.append({**rule, "antecedent_memberships": antecedents, "strength": firing})
                strengths[rule["then"]] = max(strengths[rule["then"]], firing)
        curve = np.maximum.reduce([np.minimum(self.output_curves[t], strength) for t, strength in strengths.items()])
        mass = float(curve.sum())
        if mass <= 0:
            raise ValueError("No rules cover this input; refusing a silent fallback score")
        score = float(np.dot(self.universe, curve) / mass)
        return score, {"memberships": memberships, "active_rules": active,
                       "output_strengths": strengths, "centroid": score,
                       "discrete_output_mass": mass, "universe_points": len(self.universe)}

    def score(self, query, identity):
        d = self.corpus.by_id[identity]
        features, details = self.features(query, d)
        fuzzy, inference = self.infer(features)
        weights = self.config["feature_weights"]
        contributions = {f: features[f] * weights[f] / sum(weights.values()) for f in FEATURES}
        linear = sum(contributions.values())
        alpha = self.config["alpha"]
        scores = {"bm25": details["bm25"], "linear": linear, "mamdani": fuzzy,
                  "hybrid": alpha * fuzzy + (1 - alpha) * linear}
        # A disclosed lexical evidence gate prevents quality alone ranking nonmatches.
        gate = details["bm25"] > 0
        if not gate:
            scores = {key: 0. for key in scores}
        return {"dataset_id": identity, "title": d.title, "scores": scores,
                "features": features, "evidence": details, "inference": inference,
                "linear_contributions": contributions, "alpha": alpha,
                "ungated_hybrid": alpha * fuzzy + (1 - alpha) * linear,
                "lexical_gate_passed": gate, "config_hash": self.policy_hash,
                "corpus_hash": self.corpus.hash}

    def rank(self, query, system="hybrid", candidates=None, limit=10, include_zero=False):
        if system not in {"bm25", "linear", "mamdani", "hybrid"} or limit < 1:
            raise ValueError("Invalid ranking system or cutoff")
        identities = sorted(set(candidates)) if candidates is not None else list(self.corpus.by_id)
        if any(x not in self.corpus.by_id for x in identities):
            raise ValueError("Unknown candidate dataset")
        results = [self.score(query, identity) for identity in identities]
        if not include_zero:
            results = [r for r in results if r["lexical_gate_passed"]]
        results.sort(key=lambda r: (-r["scores"][system], r["dataset_id"]))
        return [{**r, "rank": i + 1, "system": system, "score": r["scores"][system]}
                for i, r in enumerate(results[:limit])]


def explain(trace):
    e = trace["evidence"]
    lines = [f"Dataset: {trace['title']}", f"Reference date: {e['reference_date']}"]
    if e["date_status"] == "valid":
        lines.append(f"Metadata modified: {e['metadata_modified']} ({e['age_days']:.1f} days before reference).")
    else:
        lines.append(f"Metadata date is {e['date_status']}; the declared missing-date value is used.")
    lines.append(f"Resource count: {e['resource_count']} (download availability is not verified).")
    for match in e["matched_terms"]:
        lines.append(f"Matched '{match['term']}' in " + ", ".join(match["fields"]) + ".")
    lines.append(f"Mamdani centroid: {trace['inference']['centroid']:.6f}; linear score: {sum(trace['linear_contributions'].values()):.6f}.")
    lines.append(f"Hybrid = {trace['alpha']:.2f} × centroid + {1-trace['alpha']:.2f} × linear = {trace['ungated_hybrid']:.6f} before lexical gate.")
    lines.append(f"Lexical gate: {'passed' if trace['lexical_gate_passed'] else 'failed; final scores set to zero'}.")
    for rule in sorted(trace["inference"]["active_rules"], key=lambda r: (-r["strength"], r["id"])):
        lines.append(f"{rule['id']}: " + " AND ".join(f"{f}={t}" for f, t in rule["if"].items())
                     + f" -> {rule['then']}; weight={rule['weight']:.3f}, firing={rule['strength']:.6f}.")
    return "\n".join(line.rstrip() for line in lines)
