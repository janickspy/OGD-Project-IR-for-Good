import copy, math, tempfile, unittest
from pathlib import Path
from ogd_ir.model import Corpus,Dataset
from ogd_ir.config import default_config,validate
from ogd_ir.ranking import Ranker,triangle
from ogd_ir.evaluation import metrics,evaluate,import_judgments
from ogd_ir.io import read_json,write_json

def corpus():
 return Corpus([Dataset.parse(dict(id='a',title='bicycle routes',description='Historical bicycle routes in the region with documented collection methods.',publisher='P',keywords=['bicycle'],modified='2000-01-01',resource_count=3)),Dataset.parse(dict(id='b',title='image collection',modified='2026-03-06',resource_count=10))])

class Engine(unittest.TestCase):
 def test_shoulders_not_universal(self):
  self.assertEqual(float(triangle(1,[0,0,.5])),0)
  self.assertEqual(float(triangle(0,[.5,1,1])),0)
  self.assertEqual(float(triangle(0,[0,0,.5])),1)
  with self.assertRaises(ValueError): triangle(.5,[1,1,1])
 def test_coverage_and_trace_formula(self):
  r=Ranker(corpus())
  from itertools import product
  for values in product([0,.5,1],repeat=4):
   score,t=r.infer(dict(zip(('similarity','completeness','freshness','resources'),values)))
   self.assertTrue(0<=score<=1);self.assertTrue(t['active_rules'])
  t=r.score('bicycle','a');self.assertAlmostEqual(t['scores']['hybrid'],.65*t['inference']['centroid']+.35*sum(t['linear_contributions'].values()))
 def test_no_match_cannot_win_on_metadata(self):
  r=Ranker(corpus());self.assertEqual([t['dataset_id'] for t in r.rank('bicycle')],['a'])
  self.assertEqual(r.score('bicycle','b')['scores']['hybrid'],0)
  self.assertEqual(r.rank(''),[])
 def test_dates_and_determinism(self):
  r=Ranker(corpus());t=r.score('bicycle','a');self.assertGreater(t['evidence']['age_days'],9000)
  self.assertEqual(t,Ranker(corpus()).score('bicycle','a'))
  self.assertLess(t['features']['freshness'],.001)
 def test_bm25_corpus_statistics(self):
  r=Ranker(corpus());score=r.score('bicycle','a')['scores']['bm25']
  self.assertEqual(score,r.rank('bicycle',system='bm25',candidates=['a'])[0]['score'])
  self.assertEqual(len(r.index.docs),2)
 def test_metrics_hand_calculation(self):
  m=metrics(['b','a'],{'a':2,'b':0});self.assertEqual(m['map10'],.5);self.assertEqual(m['p5'],.2);self.assertAlmostEqual(m['ndcg10'],1/math.log2(3))
  with self.assertRaises(ValueError): metrics(['unknown'],{'a':1})
 def test_labels_guard_and_atomic_output(self):
  data={'status':'inherited_unverified','provenance':'test','queries':[{'id':'q','text':'bicycle','judgments':{'a':1,'b':0}}]}
  with self.assertRaises(ValueError): evaluate(Ranker(corpus()),data)
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'labels.json';write_json(path,data);before=path.read_bytes();data['queries'][0]['judgments']['a']=''
   with self.assertRaises(ValueError): import_judgments(data,corpus(),path)
   self.assertEqual(before,path.read_bytes())
 def test_invalid_policy(self):
  for value in [float('nan'),0,-1]:
   c=default_config();c['bm25']['k1']=value
   with self.assertRaises(ValueError): validate(c)
  c=default_config();c['memberships']['freshness']['low']=[1,1,1]
  with self.assertRaises(ValueError): validate(c)
 def test_config_cannot_mutate_active_ranker(self):
  cfg=default_config();r=Ranker(corpus(),cfg);before=r.score('bicycle','a');cfg['alpha']=0
  self.assertEqual(before,r.score('bicycle','a'))
 def test_future_date_is_flagged(self):
  c=Corpus([Dataset.parse(dict(id='future',title='data',modified='2099-01-01'))]);t=Ranker(c).score('data','future')
  self.assertEqual(t['evidence']['date_status'],'future');self.assertEqual(t['features']['freshness'],.5)
 def test_pool_labels_stay_unassessed(self):
  from ogd_ir.annotation import build_pool
  with tempfile.TemporaryDirectory() as td:
   data=build_pool(Ranker(corpus()),[{'id':'q','text':'bicycle'}],Path(td)/'pool.json')
   self.assertIsNone(data['queries'][0]['judgments']['a'])
   with self.assertRaises(ValueError):evaluate(Ranker(corpus()),data,allow_unverified=True)
 def test_full_corpus_requires_judgments(self):
  data={'status':'verified','provenance':'test','queries':[{'id':'q','text':'bicycle','judgments':{'b':0}}]}
  with self.assertRaises(ValueError):evaluate(Ranker(corpus()),data,mode='full_corpus')
 def test_duplicate_json(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'x';p.write_text('{"a":0,"a":1}')
   with self.assertRaises(ValueError): read_json(p)
if __name__=='__main__': unittest.main()
