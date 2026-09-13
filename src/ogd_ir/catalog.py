"""Display metadata and live CKAN normalization, separate from frozen v1 data."""
from collections import Counter
from dataclasses import asdict
from urllib.parse import urlparse, quote
from .model import Dataset, flatten
from .io import digest


def safe_url(value):
    value = str(value or '').strip()
    parsed = urlparse(value)
    return value if parsed.scheme in ('http', 'https') and parsed.netloc and not parsed.username else ''


def localized(value, language='en'):
    if isinstance(value, dict):
        value = value.get(language) or next((value.get(k) for k in ('en','de','fr','it') if value.get(k)), '')
    return flatten(value).strip()


def names(items):
    if isinstance(items, str): return [items]
    if isinstance(items, dict): return sorted(set(flatten(x).strip() for x in items.values() if x))
    return sorted(set(flatten(x.get('name') or x.get('display_name') or x.get('title')) if isinstance(x,dict)
                      else str(x) for x in (items or [])))


def catalog_entry(raw):
    organization = raw.get('organization') or {}
    resources = raw.get('resources') or []
    publisher = raw.get('publisher') or {}
    if isinstance(publisher,dict):publisher=publisher.get('name') or publisher
    return {
        'id': raw['id'], 'name': raw.get('name', ''), 'title': raw.get('title', ''),
        'description': raw.get('description') or raw.get('notes') or '',
        'publisher': raw.get('organization_title') or organization.get('title') or publisher or '',
        'organization': raw.get('organization_name') or organization.get('name') or '',
        'themes': names(raw.get('themes') or raw.get('groups')),
        'keywords': names(raw.get('keywords') or raw.get('tags')),
        'formats': sorted(set(str(x).upper() for x in raw.get('resource_formats', []) if x) |
                          {str(r['format']).upper() for r in resources if r.get('format')}),
        'license': raw.get('license_id') or raw.get('license') or '',
        'languages': names(raw.get('language')), 'created': raw.get('metadata_created'),
        'modified': raw.get('modified') or raw.get('metadata_modified'),
        'temporal_start': raw.get('temporal_coverage_start'), 'temporal_end': raw.get('temporal_coverage_end'),
        'spatial': raw.get('spatial_coverage') or raw.get('spatial'),
        'frequency': raw.get('accrual_periodicity'),
        'url': 'https://opendata.swiss/en/dataset/' + quote(raw.get('name') or raw['id'], safe=''),
        'resources': [{'id':r.get('id'), 'name':r.get('name') or r.get('description'),
                       'format':r.get('format'), 'url':safe_url(r.get('url') or r.get('download_url'))}
                      for r in resources],
        'declared_has_api': raw.get('has_api'), 'declared_has_download': raw.get('has_download'),
    }


def normalize_live(raw):
    """v2 indexes human metadata labels, never tag UUIDs/state or contact details."""
    entry = catalog_entry(raw)
    publisher = entry['organization'] or localized(entry['publisher'])
    row = dict(id=raw['id'], title=raw.get('title'), description=entry['description'],
               publisher=publisher, keywords=entry['keywords'], themes=entry['themes'],
               modified=entry['modified'], resource_count=raw.get('num_resources',len(raw.get('resources') or [])),
               license=entry['license'], url=entry['url'])
    return Dataset.parse(row), entry


class Catalog:
    def __init__(self, corpus, entries=None):
        self.corpus = corpus
        self.entries = {}
        for d in corpus.datasets:
            fallback = {**asdict(d), 'organization':d.publisher, 'formats':[], 'languages':[],
                        'themes':[d.themes] if d.themes else [], 'resources':[], 'url':safe_url(d.url)}
            self.entries[d.id] = (entries or {}).get(d.id, fallback)
        self.hash = digest(self.entries)

    def facets(self):
        result = {k:Counter() for k in ('organization','themes','formats','license','languages')}
        for entry in self.entries.values():
            for key in result:
                vals = entry.get(key) or []
                for value in ([vals] if isinstance(vals,str) else vals):
                    result[key][value] += 1
        return {k:dict(sorted(v.items())) for k,v in result.items()}

    def select(self, filters=None):
        filters = filters or {}
        unknown = set(filters) - {'organization','themes','formats','license','languages'}
        if unknown: raise ValueError(f'Unknown filters: {unknown}')
        selected = []
        for identity, entry in self.entries.items():
            def matches(key, allowed):
                values = entry.get(key) or []
                if isinstance(values,str): values = [values]
                return not allowed or bool(set(values) & set(allowed))
            if all(matches(key, vals) for key,vals in filters.items()): selected.append(identity)
        return selected
