"""Offline CLI: the same engine powers ranking, explanations and evaluation."""
import argparse
import json
from .model import Corpus
from .config import load_config
from .io import read_json, write_json
from .ranking import Ranker, explain
from .evaluation import evaluate, import_judgments

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',required=True); p.add_argument('--config')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('search'); s.add_argument('query'); s.add_argument('--system',default='hybrid'); s.add_argument('--limit',type=int,default=10); s.add_argument('--explain',action='store_true')
    e=sub.add_parser('evaluate'); e.add_argument('--judgments',required=True); e.add_argument('--output',required=True); e.add_argument('--allow-unverified',action='store_true'); e.add_argument('--mode',choices=['judged_pool','full_corpus'],default='judged_pool')
    i=sub.add_parser('import-judgments'); i.add_argument('--input',required=True); i.add_argument('--output',required=True)
    args=p.parse_args(); corpus=Corpus.load(args.corpus)
    if args.command=='import-judgments': import_judgments(read_json(args.input),corpus,args.output); return
    ranker=Ranker(corpus,load_config(args.config))
    if args.command=='evaluate': write_json(args.output,evaluate(ranker,read_json(args.judgments),allow_unverified=args.allow_unverified,mode=args.mode))
    else:
        results=ranker.rank(args.query,system=args.system,limit=args.limit)
        print('\n\n'.join(explain(r) for r in results) if args.explain else json.dumps(results,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
