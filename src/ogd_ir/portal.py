"""Read-only CKAN client; resumable snapshots and actual captured portal orders."""
from datetime import datetime, timezone
from pathlib import Path
import time
import requests
from .catalog import normalize_live
from .model import Corpus
from .io import read_json, write_json, digest

ENDPOINT = 'https://opendata.swiss/api/3/action'


class PortalError(RuntimeError): pass


class PortalClient:
    def __init__(self, session=None, timeout=30, retries=3, delay=.25):
        self.session = session or requests.Session()
        self.timeout, self.retries, self.delay = timeout, retries, delay
        self.session.headers.update({'User-Agent':'OGD-Project-IR-for-Good/0.2 (metadata research)'})

    def action(self, name, **params):
        if name not in ('package_search','package_show','organization_list','group_list'):
            raise ValueError('Unsupported read-only CKAN action')
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(f'{ENDPOINT}/{name}',params=params,timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.retries:
                        time.sleep(min(2 ** attempt,8)); continue
                response.raise_for_status()
                payload = response.json()
                if not payload.get('success') or 'result' not in payload:
                    raise PortalError('CKAN reported an unsuccessful request')
                time.sleep(self.delay)
                return payload['result']
            except (requests.Timeout,requests.ConnectionError) as exc:
                if attempt < self.retries: time.sleep(min(2 ** attempt,8)); continue
                raise PortalError(f'Portal unavailable after {attempt+1} attempts ({type(exc).__name__})') from exc
            except (requests.HTTPError,ValueError) as exc:
                status = getattr(getattr(exc,'response',None),'status_code',None)
                raise PortalError(f'Portal request failed ({status or type(exc).__name__}); no fallback data was substituted') from exc

    def search(self, query='', *, start=0, rows=100, fq='', sort='score desc, id asc'):
        if not 0 <= start or not 1 <= rows <= 1000: raise ValueError('Invalid CKAN page')
        result = self.action('package_search',q=query,rows=rows,start=start,fq=fq,sort=sort)
        if not isinstance(result.get('results'),list) or type(result.get('count')) is not int:
            raise PortalError('Unexpected CKAN search response')
        return result

    def dataset(self, identity): return self.action('package_show',id=identity)


def collect_snapshot(directory, *, client=None, query='', fq='', limit=500, page_size=100, resume=False):
    """Checkpoints before each next page. Collection is not a transactional portal dump."""
    if limit < 1 or not 1 <= page_size <= 1000: raise ValueError('Invalid collection size')
    client = client or PortalClient()
    directory = Path(directory); checkpoint = directory/'checkpoint.json'
    settings = {'endpoint':ENDPOINT,'query':query,'fq':fq,'limit':limit,'page_size':page_size,'sort':'id asc'}
    if checkpoint.exists() and not resume: raise ValueError('Snapshot exists; choose a new directory or resume it')
    state = read_json(checkpoint) if resume and checkpoint.exists() else {
        'settings':settings,'started_at':datetime.now(timezone.utc).isoformat(),'next_start':0,
        'raw_records':{},'observed_counts':[],'complete':False}
    if state['settings'] != settings: raise ValueError('Resume settings differ from checkpoint')
    while not state['complete'] and len(state['raw_records']) < limit:
        page = client.search(query,start=state['next_start'],rows=min(page_size,limit-len(state['raw_records'])),fq=fq,sort='id asc')
        records = page['results']; state['observed_counts'].append(page['count'])
        if not records and state['next_start'] < page['count']:
            raise PortalError('Unexpected empty page; checkpoint retained for retry')
        for raw in records:
            normalize_live(raw)  # reject malformed records before advancing the checkpoint
            state['raw_records'][raw['id']] = raw
        state['next_start'] += len(records)
        state['complete'] = state['next_start'] >= page['count'] or len(state['raw_records']) >= limit
        if state['complete']:state['completed_at']=datetime.now(timezone.utc).isoformat()
        write_json(checkpoint,state)
        if not records: break
    if not state['raw_records']: raise PortalError('No datasets matched the collection request')
    rows = [state['raw_records'][key] for key in sorted(state['raw_records'])]
    from dataclasses import asdict
    normalized = [normalize_live(raw) for raw in rows]
    corpus = Corpus([pair[0] for pair in normalized]); catalog = {entry['id']:entry for _,entry in normalized}
    write_json(directory/'corpus.json',[asdict(d) for d in corpus.datasets])
    write_json(directory/'catalog.json',catalog)
    manifest = {'schema_version':1,'normalization':'ckan_labels_v2','settings':settings,
                'started_at':state['started_at'],'completed_at':state['completed_at'],
                'observed_counts':state['observed_counts'],'records':len(rows),'corpus_hash':corpus.hash,
                'raw_records_hash':digest(rows),'catalog_hash':digest(catalog),
                'limited':len(rows)<max(state['observed_counts']),
                'caveat':'Portal may change during pagination; stable id order and deduplication do not establish a transactional snapshot.'}
    write_json(directory/'manifest.json',manifest)
    return manifest


def capture_portal_run(queries, output, *, depth=100, client=None, fq=''):
    if not queries or depth<1 or len({q['id'] for q in queries}) != len(queries): raise ValueError('Invalid queries/depth')
    client = client or PortalClient(); runs = {}
    for query in queries:
        ids=[]; count=None; start=0
        while len(ids)<depth:
            page=client.search(query['text'],start=start,rows=min(100,depth-len(ids)),fq=fq)
            count=page['count']; records=page['results']
            if not records: break
            ids.extend(r['id'] for r in records); start+=len(records)
            if start>=count:break
        if len(ids)!=len(set(ids)): raise PortalError('Portal changed during capture: duplicate result IDs')
        runs[query['id']]={'query':query['text'],'order':ids,'portal_count':count}
    result={'kind':'live_portal_order','endpoint':ENDPOINT,'captured_at':datetime.now(timezone.utc).isoformat(),
            'depth':depth,'fq':fq,'sort':'score desc, id asc','runs':runs,
            'caveat':'Live portal search has a different collection and time from any earlier frozen snapshot.'}
    write_json(output,result); return result
