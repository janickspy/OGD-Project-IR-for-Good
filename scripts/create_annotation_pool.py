from pathlib import Path
from ogd_ir.model import Corpus
from ogd_ir.ranking import Ranker
from ogd_ir.io import read_json
from ogd_ir.annotation import build_pool
root=Path(__file__).resolve().parents[1]
build_pool(Ranker(Corpus.load(root/'data/legacy/corpus.json')),read_json(root/'data/legacy/judgments.json')['queries'],root/'data/local/annotation_pool.json')
