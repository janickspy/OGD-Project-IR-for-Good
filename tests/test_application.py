"""Application boundaries: migration, API faults, judgment persistence and real UI flows."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import requests
from ogd_ir.model import Corpus,Dataset
from ogd_ir.config import default_config
from ogd_ir.ranking import Ranker
from ogd_ir.catalog import Catalog,normalize_live,safe_url
from ogd_ir.portal import PortalClient,PortalError,collect_snapshot,capture_portal_run
from ogd_ir.query import parse_query,llm_suggestion
from ogd_ir.calibration import propose_calibration
from ogd_ir.annotation import build_pool
from ogd_ir.storage import StudyStore
from ogd_ir.evaluation import evaluate
from ogd_ir.semantic import SemanticIndex
from ogd_ir.io import read_json,write_json,parse_json

ROOT=Path(__file__).resolve().parents[1]


def raw(identity):
    return {'id':identity,'name':'data-'+identity,'title':{'de':'Verkehr und Bevölkerung','en':'Transport and population'},
            'notes':{'en':'A documented collection of population and transport statistics for Switzerland.'},
            'organization':{'name':'publisher','title':{'en':'Publisher'}},'metadata_modified':'2026-01-01',
            'tags':[{'name':'population','id':'tag-uuid','state':'active'}],
            'groups':[{'name':'population'}],'license_id':'open',
            'resources':[{'id':'res','format':'CSV','url':'https://example.org/data.csv'}]}


@pytest.fixture
def ranker():return Ranker(Corpus([normalize_live(raw('a'))[0],Dataset.parse({'id':'b','title':'Historical employment','modified':'2000-01-01'})]))


def test_live_normalization_keeps_labels_and_resources_without_indexing_ids():
    d,entry=normalize_live(raw('a'))
    assert d.keywords=='population' and 'uuid' not in d.keywords
    assert entry['title']['de']=='Verkehr und Bevölkerung'
    assert entry['resources'][0]['url'].endswith('.csv')
    c=Catalog(Corpus([d]),{'a':entry});assert c.select({'formats':['CSV']})==['a'];assert c.select({'formats':['ZIP']})==[]
    assert safe_url('javascript:alert(1)')==''


def test_historical_ranking_inputs_and_labels_remain_unchanged():
    c=Corpus.load(ROOT/'data/legacy/corpus.json');q=read_json(ROOT/'data/legacy/judgments.json')
    previous=read_json(ROOT/'results/legacy_reanalysis.json')
    assert len(c.datasets)==500 and c.hash==previous['corpus_hash']
    assert len(q['queries'])==15 and sum(len(x['judgments']) for x in q['queries'])==150
    catalog=read_json(ROOT/'data/legacy/catalog.json');assert set(catalog)==set(c.by_id)
    assert all('contact_points' not in row for row in catalog.values())


class Response:
    def __init__(self,status,payload):self.status_code=status;self.payload=payload
    def json(self):return self.payload
    def raise_for_status(self):
        if self.status_code>=400:raise requests.HTTPError(response=self)


class Session:
    def __init__(self,results):self.headers={};self.results=iter(results);self.calls=[]
    def get(self,*args,**kwargs):
        self.calls.append(kwargs);result=next(self.results)
        if isinstance(result,Exception):raise result
        return result


def test_api_retries_transient_errors_and_reports_forbidden(monkeypatch):
    monkeypatch.setattr('ogd_ir.portal.time.sleep',lambda _:None)
    session=Session([Response(503,{}),requests.Timeout(),Response(200,{'success':True,'result':{'count':1,'results':[raw('a')]}})])
    assert PortalClient(session=session).search()['count']==1
    assert len(session.calls)==3
    with pytest.raises(PortalError,match='403'):
        PortalClient(session=Session([Response(403,{})])).search()


def test_snapshot_resumes_without_discarding_first_page(tmp_path):
    class Client:
        def __init__(self,fail):self.fail=fail;self.starts=[]
        def search(self,*args,**kwargs):
            start=kwargs['start'];self.starts.append(start)
            if start==1 and self.fail:raise PortalError('network interrupted')
            return {'count':3,'results':[raw(str(start))]}
    with pytest.raises(PortalError):collect_snapshot(tmp_path,client=Client(True),limit=3,page_size=1)
    assert read_json(tmp_path/'checkpoint.json')['next_start']==1
    resumed=Client(False);manifest=collect_snapshot(tmp_path,client=resumed,limit=3,page_size=1,resume=True)
    assert resumed.starts==[1,2] and manifest['records']==3
    assert len(Corpus.load(tmp_path/'corpus.json').datasets)==3
    with pytest.raises(ValueError,match='settings'):collect_snapshot(tmp_path,client=resumed,limit=4,page_size=1,resume=True)


def test_portal_order_is_captured_not_approximated(tmp_path):
    class Client:
        def search(self,*a,**kw):return {'count':2,'results':[raw('b'),raw('a')]}
    result=capture_portal_run([{'id':'Q','text':'population'}],tmp_path/'portal.json',client=Client())
    assert result['runs']['Q']['order']==['b','a'] and result['kind']=='live_portal_order'


def test_query_expansion_is_explicit_and_preserves_historical_intent(monkeypatch):
    plain=parse_query('  historical population 1990  ',language='en')
    expanded=parse_query('historical population 1990',language='en',expand=True)
    assert plain['search_text']=='historical population 1990'
    assert expanded['temporal_intent']=='historical' and expanded['years']==['1990']
    assert 'bevölkerung' in expanded['expansion_terms']
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    with pytest.raises(ValueError,match='OPENAI_API_KEY'):llm_suggestion('data',model='test')


def test_constant_metadata_never_creates_degenerate_calibration():
    c=Corpus([normalize_live(raw(str(i)))[0] for i in range(8)])
    ranker=Ranker(c);before=deepcopy(ranker.config);proposal=propose_calibration(ranker,['population'])
    assert ranker.config==before
    assert all(not item['changed'] for item in proposal['decisions'].values())
    assert proposal['config_hash']==ranker.policy_hash


def test_assessment_persistence_disagreement_and_stale_adjudication(tmp_path,ranker):
    pool=build_pool(ranker,[{'id':'Q','text':'population'}]);s=StudyStore(tmp_path/'study.sqlite');pid=s.add_pool(pool)
    with pytest.raises(ValueError):s.rate(pid,'Q','a','one',None)
    s.rate(pid,'Q','a','one',2);s.rate(pid,'Q','a','two',0)
    assert StudyStore(tmp_path/'study.sqlite').ratings(pid,'one')[0]['grade']==2
    with pytest.raises(ValueError,match='Resolve disagreement'):s.export(pid,attested_by='lead')
    s.adjudicate(pid,'Q','a','lead',1,'Some relevant content');assert s.export(pid,attested_by='lead')['queries'][0]['judgments']['a']==1
    s.rate(pid,'Q','a','two',1)
    with pytest.raises(ValueError,match='Resolve disagreement'):s.export(pid,attested_by='lead')
    assert s.agreement(pid,'one','two')['exact_agreement']==0
    with pytest.raises(ValueError):s.agreement(pid,'one','one')


class Encoder:
    provenance={'model':'explicit-test-double','revision':'test'}
    def encode(self,texts):
        return np.array([[1.,0.] if 'population' in t.lower() else [0.,1.] for t in texts],dtype=np.float32)


def test_semantic_pool_and_eval_have_same_candidates_and_separate_provenance(tmp_path,ranker):
    dense=SemanticIndex(ranker.corpus,Encoder(),tmp_path)
    assert dense.rank('population')[0]['dataset_id']=='a'
    pool=build_pool(ranker,[{'id':'Q','text':'population'}],depth=2,semantic=dense)
    assert set(pool['queries'][0]['judgments'])=={'a','b'}
    pool['queries'][0]['judgments']={'a':2,'b':0};pool['status']='verified'
    result=evaluate(ranker,pool,semantic=dense)
    assert len(result['paired_tests'])==10 and result['semantic']['encoder']==Encoder.provenance
    assert set(result['runs']['Q']['semantic'])=={'a','b'}
    assert SemanticIndex(ranker.corpus,Encoder(),tmp_path).key==dense.key


def test_upload_json_is_strict():
    with pytest.raises(ValueError):parse_json('{"grade":2,"grade":0}')
    with pytest.raises(ValueError):parse_json('{"grade":NaN}')


def test_streamlit_search_filters_and_all_workspaces(tmp_path,monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('OGD_WORKSPACE',str(tmp_path))
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=45).run()
    assert not app.exception and not app.error
    app.button(key='FormSubmitter:query_form-Search').click().run()
    assert not app.exception and not app.error and len(app.subheader)>=1
    methods=next(x for x in app.selectbox if x.label=='Ranking method')
    methods.set_value('bm25').run();assert not app.error
    page=next(x for x in app.number_input if x.label=='Page')
    if page.max>1:page.set_value(2).run();assert not app.error
    for workspace in ('Collection','Analytics','Assessments','Evaluation','Calibration','Study'):
        app.sidebar.radio[0].set_value(workspace).run()
        assert not app.exception and not app.error, workspace
    app.sidebar.radio[0].set_value('Assessments').run()
    next(b for b in app.button if b.label=='Create blind pool').click().run()
    assert not app.error and any(x.label=='Assessor pseudonym' for x in app.text_input)


def test_study_counterbalance_and_immutable_responses(tmp_path):
    from ogd_ir.study import make_protocol,assignment,StudySession,CONDITIONS
    c=Corpus([normalize_live(raw(str(i)))[0] for i in range(3)])
    protocol=make_protocol(Ranker(c),[{'id':str(i),'text':q} for i,q in enumerate(['population','transport','Verkehr'])])
    for slot in range(3):assert {t['condition'] for t in assignment(protocol,slot)}==set(CONDITIONS)
    for position in range(3):assert len({assignment(protocol,slot)[position]['condition'] for slot in range(3)})==3
    trial=assignment(protocol,0)[0];s=StudySession(StudyStore(tmp_path/'db.sqlite'))
    s.start(protocol,'P01',0,trial['task']['id'],trial['condition'])
    with pytest.raises(ValueError):s.complete(protocol,'P01',trial['task']['id'],[None,1,1],3)
    s.complete(protocol,'P01',trial['task']['id'],[0,1,1],4)
    with pytest.raises(ValueError,match='overwritten'):s.complete(protocol,'P01',trial['task']['id'],[0,0,0],1)
    assert s.records(protocol)[0]['responses']['correct']==[True,True,True]


def test_blind_pool_order_survives_json_persistence(tmp_path,ranker):
    from ogd_ir.semantic import SemanticIndex
    pool=build_pool(ranker,[{'id':'Q','text':'population'}],depth=2,semantic=SemanticIndex(ranker.corpus,Encoder()))
    s=StudyStore(tmp_path/'db.sqlite');pid=s.add_pool(pool)
    assert s.pool(pid)['queries'][0]['candidate_order']==pool['queries'][0]['candidate_order']


def test_streamlit_study_collects_only_submitted_observations(tmp_path,monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('OGD_WORKSPACE',str(tmp_path))
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=45).run()
    app.sidebar.radio[0].set_value('Study').run()
    next(b for b in app.button if b.label=='Create study protocol').click().run()
    assert not app.error and not app.exception
    next(c for c in app.checkbox if c.label.startswith('I agree')).check()
    next(t for t in app.text_input if t.label=='Participant pseudonym').set_value('TEST-ONLY').run()
    next(b for b in app.button if b.label=='Start this task').click().run()
    for radio,answer in zip([r for r in app.radio if r.label!='Workspace'],[0,1,1]):radio.set_value(answer)
    next(s for s in app.selectbox if s.label.startswith('How clear')).set_value(4)
    next(b for b in app.button if b.label=='Submit task answers').click().run()
    assert not app.error and not app.exception
    from ogd_ir.study import StudySession
    rows=StudySession(StudyStore(tmp_path/'workbench.sqlite')).records(read_json(tmp_path/'study/protocol.json'))
    assert len(rows)==1 and rows[0]['responses']['correct']==[True,True,True]
