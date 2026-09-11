"""Fetch frozen historical inputs, verify bytes, then normalize. No code reuse."""
import hashlib
from pathlib import Path
import urllib.request
from dataclasses import asdict
from ogd_ir.io import read_json,write_json
from ogd_ir.model import Corpus
from ogd_ir.config import default_config
from ogd_ir.ranking import rules
ROOT=Path(__file__).resolve().parents[1]
COMMIT='307d454248182cf4221a15d5853ae337da7c7798'
SOURCES={'corpus':'data/raw/ogd_metadata_20260306_183841.json','qrels':'evaluation/ground_truth_final.json'}

def main():
    manifest=read_json(ROOT/'data/legacy/source_manifest.json')
    cache=ROOT/'data/local'; cache.mkdir(parents=True,exist_ok=True)
    for key,path in SOURCES.items():
        target=cache/(key+'.json')
        if not target.exists():
            target.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/Deep0901/Master_thesis_project/{COMMIT}/{path}',timeout=60).read())
        if hashlib.sha256(target.read_bytes()).hexdigest()!=manifest['files'][key]['sha256']:
            raise ValueError('Source checksum mismatch: '+key)
    corpus=Corpus.load(cache/'corpus.json')
    write_json(ROOT/'data/legacy/corpus.json',[asdict(d) for d in corpus.datasets])
    raw=read_json(cache/'qrels.json')
    queries=[]
    for qid,q in sorted(raw.items()):
        ids=[j['dataset_id'] for j in q['judgments']]
        if len(set(ids))!=len(ids): raise ValueError('Duplicate historical judgments')
        queries.append({'id':qid,'text':q['query']['query_text'], 'judgments':{j['dataset_id']:j['relevance'] for j in q['judgments']}})
    write_json(ROOT/'data/legacy/judgments.json',{'status':'inherited_unverified','provenance':manifest,'queries':queries})
    write_json(ROOT/'configs/default.json',default_config())
    write_json(ROOT/'configs/rules.json',rules(default_config()))
if __name__=='__main__': main()
