"""Run a declared five-system comparison, including actual local ONNX inference."""
import argparse
from pathlib import Path
from ogd_ir.model import Corpus
from ogd_ir.ranking import Ranker
from ogd_ir.config import load_config
from ogd_ir.semantic import MiniLMEncoder,SemanticIndex
from ogd_ir.evaluation import evaluate
from ogd_ir.io import read_json,write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-cache',default=str(ROOT/'data/local/models'))
    p.add_argument('--offline',action='store_true')
    args=p.parse_args()
    encoder=MiniLMEncoder(args.model_cache,offline=args.offline)
    for filename,output in [('corpus.json','semantic_reanalysis.json'),('corpus_v2.json','application_reanalysis.json')]:
        corpus=Corpus.load(ROOT/'data/legacy'/filename)
        dense=SemanticIndex(corpus,encoder,ROOT/'data/local/embeddings')
        result=evaluate(Ranker(corpus,load_config(ROOT/'configs/default.json')),
                        read_json(ROOT/'data/legacy/judgments.json'),allow_unverified=True,semantic=dense)
        write_json(ROOT/'results'/output,result)
        print(filename,{system:round(row['ndcg10']['mean'],6) for system,row in result['aggregates']['all_queries'].items()})


if __name__=='__main__':main()
