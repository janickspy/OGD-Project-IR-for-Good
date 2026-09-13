"""Search, collect, assess and evaluate through the same application services."""
import argparse
import json
from .model import Corpus
from .config import load_config
from .io import read_json, write_json
from .ranking import Ranker, explain
from .evaluation import evaluate, import_judgments


def queries(path):
    data=read_json(path)
    return data['queries'] if isinstance(data,dict) else data


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',default='data/legacy/corpus.json');p.add_argument('--config')
    p.add_argument('--model-cache',default='data/local/models');p.add_argument('--embedding-cache',default='data/local/embeddings')
    p.add_argument('--offline-model',action='store_true')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('search');s.add_argument('query');s.add_argument('--system',choices=['bm25','linear','mamdani','hybrid','semantic'],default='hybrid')
    s.add_argument('--limit',type=int,default=10);s.add_argument('--explain',action='store_true');s.add_argument('--expand',action='store_true');s.add_argument('--language',choices=['auto','de','fr','it','en'],default='auto')
    e=sub.add_parser('evaluate');e.add_argument('--judgments',required=True);e.add_argument('--output',required=True)
    e.add_argument('--allow-unverified',action='store_true');e.add_argument('--semantic',action='store_true');e.add_argument('--mode',choices=['judged_pool','full_corpus'],default='judged_pool')
    i=sub.add_parser('import-judgments');i.add_argument('--input',required=True);i.add_argument('--output',required=True)
    c=sub.add_parser('collect');c.add_argument('--output',required=True);c.add_argument('--query',default='');c.add_argument('--fq',default='')
    c.add_argument('--limit',type=int,default=500);c.add_argument('--page-size',type=int,default=100);c.add_argument('--resume',action='store_true')
    capture=sub.add_parser('capture-portal');capture.add_argument('--queries',required=True);capture.add_argument('--output',required=True);capture.add_argument('--depth',type=int,default=100)
    pool=sub.add_parser('pool');pool.add_argument('--queries',required=True);pool.add_argument('--output',required=True);pool.add_argument('--depth',type=int,default=20);pool.add_argument('--semantic',action='store_true')
    cal=sub.add_parser('calibrate');cal.add_argument('--queries',required=True);cal.add_argument('--output',required=True)
    audit=sub.add_parser('analyse');audit.add_argument('--query');audit.add_argument('--output',required=True)
    sub.add_parser('index-semantic')
    args=p.parse_args(argv)
    try:
        if args.command=='collect':
            from .portal import collect_snapshot
            result=collect_snapshot(args.output,query=args.query,fq=args.fq,limit=args.limit,page_size=args.page_size,resume=args.resume)
            print(json.dumps(result,indent=2));return
        if args.command=='capture-portal':
            from .portal import capture_portal_run
            capture_portal_run(queries(args.queries),args.output,depth=args.depth);return
        corpus=Corpus.load(args.corpus)
        if args.command=='import-judgments':import_judgments(read_json(args.input),corpus,args.output);return
        ranker=Ranker(corpus,load_config(args.config));dense=None
        if args.command=='index-semantic' or getattr(args,'semantic',False) or getattr(args,'system',None)=='semantic':
            from .semantic import SemanticIndex,MiniLMEncoder
            dense=SemanticIndex(corpus,MiniLMEncoder(args.model_cache,offline=args.offline_model),args.embedding_cache)
        if args.command=='index-semantic':print(json.dumps(dense.metadata,indent=2));return
        if args.command=='evaluate':write_json(args.output,evaluate(ranker,read_json(args.judgments),allow_unverified=args.allow_unverified,mode=args.mode,semantic=dense))
        elif args.command=='pool':
            from .annotation import build_pool
            build_pool(ranker,queries(args.queries),args.output,depth=args.depth,semantic=dense)
        elif args.command=='calibrate':
            from .calibration import propose_calibration
            write_json(args.output,propose_calibration(ranker,[q['text'] for q in queries(args.queries)]))
        elif args.command=='analyse':
            from .analytics import corpus_summary,ranking_changes,policy_ablation
            result=corpus_summary(ranker)
            if args.query:result['ranking_changes']=ranking_changes(ranker,args.query);result['ablation']=policy_ablation(ranker,args.query)
            write_json(args.output,result)
        else:
            from .query import parse_query
            query=parse_query(args.query,args.language,args.expand)
            results=dense.rank(query['search_text'],limit=args.limit) if dense else ranker.rank(query['search_text'],system=args.system,limit=args.limit)
            if args.explain and not dense:print('\n\n'.join(explain(r) for r in results))
            else:print(json.dumps({'query':query,'results':results},indent=2,ensure_ascii=False))
    except (ValueError,RuntimeError,OSError) as exc:p.exit(2,f'Error: {exc}\n')


if __name__=='__main__':main()
