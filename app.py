"""Local OGD workbench. Run with: streamlit run app.py"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ogd_ir.io import read_json, write_json, digest, parse_json
from ogd_ir.model import Corpus
from ogd_ir.catalog import Catalog, localized, safe_url
from ogd_ir.config import load_config, FEATURES, validate
from ogd_ir.ranking import Ranker, explain, triangle
from ogd_ir.query import parse_query, llm_suggestion
from ogd_ir.portal import PortalClient, PortalError, collect_snapshot, capture_portal_run
from ogd_ir.analytics import corpus_summary, ranking_changes, policy_ablation
from ogd_ir.calibration import propose_calibration
from ogd_ir.annotation import build_pool
from ogd_ir.evaluation import evaluate
from ogd_ir.storage import StudyStore

ROOT=Path(__file__).resolve().parent
LOCAL=Path(os.environ.get('OGD_WORKSPACE',ROOT/'data/local')).resolve()
st.set_page_config(page_title='OGD · Dataset workbench',page_icon='◈',layout='wide')


def download(label,value,name,key=None):
    st.download_button(label,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),name,'application/json',key=key)


@st.cache_resource
def engine(path,content_hash,config_json):
    return Ranker(Corpus.load(path),json.loads(config_json))


@st.cache_resource
def semantic_index(path,content_hash):
    from ogd_ir.semantic import SemanticIndex, MiniLMEncoder
    return SemanticIndex(Corpus.load(path),MiniLMEncoder(cache_dir=LOCAL/'models'),LOCAL/'embeddings')


def semantic(ranker,path):
    if not st.session_state.get('semantic_enabled'):return None
    with st.spinner('Loading the pinned multilingual model and snapshot embeddings…'):
        return semantic_index(str(path),ranker.corpus.hash)


def render_card(row,ranker,catalog,language,show_explanation=True,key_prefix='search',feedback=False):
    identity=row['dataset_id'];d=ranker.corpus.by_id[identity];meta=catalog.entries[identity]
    with st.container(border=True):
        left,right=st.columns([5,1])
        left.subheader(f"{row.get('rank','')} · {localized(meta['title'],language) or d.title}")
        right.metric(row['system'].title(),f"{row['score']:.4f}")
        st.caption(' · '.join(x for x in [localized(meta.get('publisher'),language) or d.publisher,
                      f"{d.resource_count} resources",', '.join(meta.get('formats',[])),d.modified or 'Date missing'] if x))
        description=localized(meta.get('description'),language) or d.description
        st.write(description[:650]+('…' if len(description)>650 else ''))
        url=safe_url(meta.get('url'))
        if url:st.link_button('Open dataset on opendata.swiss',url)
        if show_explanation:
            with st.expander('Why this result?'):
                if row['system']=='semantic':
                    st.write(row['explanation']);st.json(row['semantic_provenance'],expanded=False)
                else:
                    st.write('Matched terms and fields')
                    matches=row['evidence']['matched_terms']
                    st.dataframe([{'Term':m['term'],'Fields':', '.join(m['fields']),'BM25 contribution':m['bm25_contribution']} for m in matches],hide_index=True)
                    st.bar_chart(pd.DataFrame({'Normalized feature':row['features']}))
                    st.text(explain(row))
                    download('Download exact scoring trace',row,f'{identity}-trace.json',key=f'{key_prefix}-trace-{identity}')
        with st.expander('Metadata and resources'):
            st.write('Temporal coverage:',meta.get('temporal_start') or 'Not supplied','—',meta.get('temporal_end') or 'Not supplied')
            st.write('Licence:',meta.get('license') or 'Not supplied')
            st.write('Update frequency:',meta.get('frequency') or 'Not supplied')
            for resource in meta.get('resources',[]):
                resource_url=safe_url(resource.get('url'))
                if resource_url:st.link_button(localized(resource.get('name'),language) or resource.get('format') or 'Resource',resource_url)
            if not meta.get('resources'):st.caption('Resource URLs are not included in this snapshot. Use the portal link.')
        if feedback:
            st.caption('Your selection saves this query, dataset and ranking configuration locally.')
            cols=st.columns(2)
            for col,label,value in [(cols[0],'Useful','useful'),(cols[1],'Not useful','not_useful')]:
                if col.button(label,key=f'{key_prefix}-feedback-{value}-{identity}'):
                    StudyStore(LOCAL/'workbench.sqlite').feedback({'rating':value,'dataset_id':identity,
                        'query':st.session_state.get('search_query',''),'system':row['system'],
                        'config_hash':ranker.policy_hash,'corpus_hash':ranker.corpus.hash})
                    st.toast('Feedback saved locally')


def search_page(ranker,catalog,path,language,filters):
    st.title('Find and understand public data')
    st.write('Search Swiss datasets, compare ranking priorities and inspect the evidence behind each result.')
    with st.form('query_form'):
        text=st.text_input('Search query',value=st.session_state.get('search_query','Verkehrsunfälle Statistik Schweiz'))
        a,b=st.columns(2)
        query_language=a.selectbox('Query language',['auto','de','fr','it','en'])
        expand=b.checkbox('Add multilingual terms from the built-in vocabulary',value=False)
        submitted=st.form_submit_button('Search',type='primary')
    if submitted:
        parsed=parse_query(text,query_language,expand)
        st.session_state.search_query=parsed['search_text'];st.session_state.query_analysis=parsed
        st.session_state.result_page=1
    with st.expander('Optional query assistance'):
        st.caption('This sends only the entered query to OpenAI when you press Suggest. Review and paste the suggestion into Search to use it.')
        model=st.text_input('OpenAI model',value=os.environ.get('OGD_OPENAI_MODEL',''))
        if st.button('Suggest a query'):
            result=llm_suggestion(text,model=model);st.code(result['suggestion'],language=None)
    query=st.session_state.get('search_query')
    if not query:st.info('Enter a query to explore this snapshot.');return
    with st.expander('Query interpretation'):
        st.json(st.session_state.get('query_analysis',{}))
        st.caption('Query assistance never changes freshness preferences or treats metadata modification as temporal coverage.')
    candidates=catalog.select(filters)
    traces=ranker.rank(query,candidates=candidates,limit=len(ranker.corpus.datasets)) if candidates else []
    orders={system:sorted(traces,key=lambda t:(-t['scores'][system],t['dataset_id'])) for system in ('hybrid','mamdani','linear','bm25')}
    dense=semantic(ranker,path)
    if dense:orders['semantic']=dense.rank(query,candidates=candidates,limit=max(1,len(candidates)))
    system=st.selectbox('Ranking method',list(orders),format_func=lambda x:{'hybrid':'Hybrid: fuzzy + linear','mamdani':'Mamdani: fuzzy rules','linear':'Linear: weighted criteria','bm25':'BM25: lexical relevance','semantic':'Semantic: multilingual MiniLM'}[x])
    rows=orders[system]
    st.caption(f"{len(rows)} results · {len(candidates)} filtered candidates · reference date {ranker.config['reference_date']}")
    st.caption('Method scores use different scales. Compare result order, not score magnitudes across methods.')
    if not rows:st.info('No matches. Try fewer filters, a different language, multilingual expansion or semantic search.');return
    with st.expander('Compare ranking methods'):
        comparison=[]
        for name,order in orders.items():
            comparison.extend({'method':name,'rank':i+1,'dataset':localized(catalog.entries[t['dataset_id']]['title'],language)} for i,t in enumerate(order[:10]))
        st.dataframe(comparison,hide_index=True)
        if traces:
            figure=go.Figure()
            for trace in orders['hybrid'][:3]:
                values=[trace['features'][f] for f in FEATURES]
                figure.add_trace(go.Scatterpolar(r=values+[values[0]],theta=list(FEATURES)+[FEATURES[0]],fill='toself',name=localized(catalog.entries[trace['dataset_id']]['title'],language)[:55]))
            figure.update_layout(polar={'radialaxis':{'range':[0,1]}},height=380,margin=dict(t=25,b=25))
            st.plotly_chart(figure,width='stretch')
    a,b=st.columns(2)
    size=a.selectbox('Results per page',[10,20,50]);pages=max(1,(len(rows)+size-1)//size)
    if st.session_state.get('result_page',1)>pages:st.session_state.result_page=pages
    page=b.number_input('Page',min_value=1,max_value=pages,step=1,key='result_page')
    show=st.checkbox('Show explanation controls',value=True)
    feedback=st.checkbox('Enable local usefulness feedback',value=False)
    export=[]
    for i,row in enumerate(rows):
        export.append({**row,'rank':i+1,'system':system,'score':row['score'] if system=='semantic' else row['scores'][system]})
    download('Download ranked results',{'query':query,'system':system,'filters':filters,'config':ranker.config,
             'corpus_hash':ranker.corpus.hash,'catalog_hash':catalog.hash,'results':export},'search-results.json')
    for row in export[(int(page)-1)*size:int(page)*size]:render_card(row,ranker,catalog,language,show,feedback=feedback)


def collection_page():
    st.title('Collect a snapshot')
    st.write('Save live portal metadata with a checkpoint, collection manifest and exact corpus hash.')
    with st.form('collect'):
        name=st.text_input('Snapshot name',value='swiss-'+datetime.now(timezone.utc).strftime('%Y%m%d'))
        query=st.text_input('Portal query (blank collects all topics)')
        fq=st.text_input('CKAN filter expression (optional)',placeholder='organization:bundesamt-fur-statistik-bfs')
        count=st.number_input('Maximum datasets',min_value=1,max_value=100000,value=500,step=100)
        resume=st.checkbox('Resume this snapshot with identical settings')
        submitted=st.form_submit_button('Collect from opendata.swiss',type='primary')
    if submitted:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',name):raise ValueError('Use only letters, numbers, hyphens and underscores for the snapshot name')
        with st.spinner('Collecting metadata; progress is checkpointed after each page…'):
            manifest=collect_snapshot(LOCAL/'snapshots'/name,query=query,fq=fq,limit=int(count),resume=resume)
        st.success('Snapshot saved. Select it in the sidebar.');st.json(manifest)
    st.caption('A live collection can change during paging. The manifest records observed counts and whether collection was limited.')
    st.subheader('Capture the portal’s actual ranking')
    portal_query=st.text_input('Query to capture')
    if st.button('Capture portal order'):
        result=capture_portal_run([{'id':'query-1','text':parse_query(portal_query)['normalized']}],LOCAL/'portal-run.json')
        download('Download captured order',result,'portal-run.json');st.json(result,expanded=False)


def analytics_page(ranker,catalog):
    st.title('Inspect the collection')
    report=corpus_summary(ranker)
    a,b,c=st.columns(3);a.metric('Datasets',report['datasets']);b.metric('Publishers',len(report['publishers']));c.metric('Usable metadata dates',report['date_status'].get('valid',0))
    st.caption(report['caveat'])
    left,right=st.columns(2)
    left.subheader('Publisher representation');left.bar_chart(pd.DataFrame.from_dict(report['publishers'],orient='index',columns=['Datasets']))
    right.subheader('Missing metadata');right.bar_chart(pd.DataFrame.from_dict(report['missing_fields'],orient='index',columns=['Datasets']))
    st.dataframe(report['rows'],hide_index=True);download('Download collection audit',report,'corpus-audit.json')
    st.subheader('Who gains or loses visibility?')
    query=st.text_input('Query for ranking analysis',value='Verkehrsunfälle Statistik Schweiz')
    if st.button('Analyse ranking changes'):
        result=ranking_changes(ranker,query);st.dataframe(result['movements'],hide_index=True)
        st.bar_chart(pd.DataFrame(result['exposure']).fillna(0));st.caption(result['scope'])
        st.json(policy_ablation(ranker,query),expanded=False)
        download('Download visibility analysis',result,'ranking-changes.json')
    with st.expander('Local feedback'):
        rows=StudyStore(LOCAL/'workbench.sqlite').feedback_rows()
        st.write(f'{len(rows)} saved feedback events');download('Export feedback',rows,'feedback.json')


def assessment_page(ranker,catalog,path,language):
    st.title('Assess relevance')
    st.write('Build a blind pool, collect independent grades and resolve disagreements before exporting verified judgments.')
    store=StudyStore(LOCAL/'workbench.sqlite')
    with st.expander('Create an assessment pool'):
        text=st.text_area('One query per line',value='Verkehrsunfälle Statistik Schweiz')
        depth=st.number_input('Pool depth per method',min_value=1,max_value=100,value=10)
        if st.button('Create blind pool'):
            queries=[{'id':f'Q{i+1:03}','text':q.strip()} for i,q in enumerate(text.splitlines()) if q.strip()]
            pool=build_pool(ranker,queries,depth=int(depth),semantic=semantic(ranker,path));identity=store.add_pool(pool)
            st.success(f"Created {sum(len(q['judgments']) for q in pool['queries'])} query–dataset pairs.")
    available=store.pools()
    if not available:st.info('Create a pool to begin.');return
    identity=st.selectbox('Pool',[r['id'] for r in available],format_func=lambda x:x[:16])
    pool=store.pool(identity)
    if pool['provenance']['corpus_hash']!=ranker.corpus.hash:st.warning('Select the snapshot used to create this pool.');return
    mode=st.radio('Workflow',['Independent assessment','Agreement and adjudication','Verified export'],horizontal=True)
    assessor=st.text_input('Assessor pseudonym')
    if mode=='Verified export':
        attested=st.checkbox('I confirm the independent assessment and adjudication process has been reviewed')
        if st.button('Prepare verified judgments',disabled=not attested):
            result=store.export(identity,attested_by=assessor);download('Download verified judgments',result,'verified-judgments.json')
        return
    query=st.selectbox('Query',pool['queries'],format_func=lambda q:q['id']+' · '+q['text'])
    ids=query.get('candidate_order',list(query['judgments']))
    identity_d=st.selectbox('Dataset (shuffled pool order)',ids,format_func=lambda i:localized(catalog.entries[i]['title'],language))
    meta=catalog.entries[identity_d]
    st.subheader(localized(meta['title'],language));st.write(localized(meta.get('description'),language))
    url=safe_url(meta.get('url'))
    if url:st.link_button('Inspect original dataset',url)
    st.caption('0 = not relevant; 1 = partially relevant; 2 = directly relevant. Leave unassessed if you cannot decide.')
    if mode=='Independent assessment':
        own=store.ratings(identity,assessor.strip()) if assessor.strip() else []
        row=next((r for r in own if r['query']==query['id'] and r['dataset']==identity_d),None)
        st.caption(f'{len(own)} saved assessments by this pseudonym in this pool.')
        with st.form(f'rating-{identity}-{query["id"]}-{identity_d}-{assessor}'):
            grade=st.selectbox('Relevance grade',[None,0,1,2],index=[None,0,1,2].index(row['grade'] if row else None),format_func=lambda x:'Unassessed' if x is None else str(x))
            note=st.text_area('Assessment note',value=row['note'] if row else '')
            if st.form_submit_button('Save assessment'):
                store.rate(identity,query['id'],identity_d,assessor,grade,note);st.success('Assessment saved.')
    else:
        ratings=store.ratings(identity);assessors=sorted({r['assessor'] for r in ratings})
        if len(assessors)<2:st.info('At least two assessors are needed.');return
        pair=st.multiselect('Agreement between assessors',assessors,default=assessors[:2],max_selections=2)
        if len(pair)==2:st.json(store.agreement(identity,*pair))
        st.dataframe([r for r in ratings if r['query']==query['id'] and r['dataset']==identity_d],hide_index=True)
        grade=st.selectbox('Adjudicated grade',[None,0,1,2],format_func=lambda x:'Choose a grade' if x is None else str(x))
        note=st.text_area('Adjudication rationale')
        if st.button('Save adjudication'):store.adjudicate(identity,query['id'],identity_d,assessor,grade,note);st.success('Adjudication saved.')


def evaluation_page(ranker,path):
    st.title('Evaluate ranking methods')
    uploaded=st.file_uploader('Judgments JSON (or use the inherited example)',type=['json'])
    data=parse_json(uploaded.getvalue()) if uploaded else read_json(ROOT/'data/legacy/judgments.json')
    st.caption(f"Label status: {data.get('status','missing')}. Blank and unjudged results cannot be treated as non-relevant.")
    allow=st.checkbox('Allow reanalysis of inherited, unverified labels',value=False)
    mode=st.selectbox('Evaluation scope',['judged_pool','full_corpus'])
    if st.button('Run evaluation',type='primary'):
        with st.spinner('Evaluating all methods on the same snapshot and policy…'):
            result=evaluate(ranker,data,allow_unverified=allow,mode=mode,semantic=semantic(ranker,path))
        st.session_state.evaluation_result=result
    result=st.session_state.get('evaluation_result')
    if result and result['corpus_hash']==ranker.corpus.hash and result['config_hash']==ranker.policy_hash:
        for subset,values in result['aggregates'].items():
            st.subheader(subset.replace('_',' ').title())
            st.dataframe([{'system':system,'n':row['n'],**{m:v['mean'] for m,v in row.items() if m!='n'}} for system,row in values.items()],hide_index=True)
        st.dataframe(result['per_query'],hide_index=True);st.dataframe(result['paired_tests'],hide_index=True)
        st.caption(result['tests_policy']);download('Download full evaluation with uncertainty and provenance',result,'evaluation.json')


def calibration_page(ranker):
    st.title('Inspect and calibrate the ranking policy')
    st.write('The current policy remains active until you explicitly adopt a proposal. Every result records its configuration hash.')
    import numpy as np
    x=np.linspace(0,1,201)
    for feature in FEATURES:
        figure=go.Figure([go.Scatter(x=x,y=triangle(x,points),name=term) for term,points in ranker.config['memberships'][feature].items()])
        figure.update_layout(title=feature.title(),height=240,margin=dict(t=40,b=20),yaxis_range=[0,1])
        st.plotly_chart(figure,width='stretch')
    download('Download active configuration',ranker.config,'ranking-config.json')
    queries=st.text_area('Representative calibration queries (one per line)',value='Verkehrsunfälle Statistik Schweiz\nBevölkerung Schweiz')
    if st.button('Propose calibration'):
        st.session_state.proposal=propose_calibration(ranker,[q.strip() for q in queries.splitlines() if q.strip()])
    proposal=st.session_state.get('proposal')
    if proposal and proposal['parent_config_hash']==ranker.policy_hash:
        st.json(proposal['decisions']);st.warning(proposal['warning']);download('Download proposal and provenance',proposal,'calibration-proposal.json')
        if st.button('Adopt this proposal for this session'):
            st.session_state.policy=proposal['config'];st.rerun()


def study_page(ranker,catalog,language):
    from ogd_ir.study import make_protocol,assignment,StudySession
    st.title('Explanation comprehension study')
    store=StudyStore(LOCAL/'workbench.sqlite');study=StudySession(store)
    protocol_path=LOCAL/'study/protocol.json'
    participant_mode=os.environ.get('OGD_STUDY_MODE')=='participant'
    if not participant_mode:
        st.caption('Facilitator workspace. Review tasks and consent procedures before recruitment. Set OGD_STUDY_MODE=participant to hide setup and response exports during sessions.')
        with st.expander('Prepare a fixed study protocol'):
            text=st.text_area('Study queries, one per line',value='Verkehr\nBevölkerung\nStatistik')
            if st.button('Create study protocol'):
                protocol=make_protocol(ranker,[{'id':f'T{i+1:03}','text':q.strip()} for i,q in enumerate(text.splitlines()) if q.strip()])
                write_json(protocol_path,protocol);st.success('Protocol saved; query order and conditions will be counterbalanced by participant slot.')
    if not protocol_path.exists():st.info('The facilitator needs to prepare a protocol first.');return
    protocol=read_json(protocol_path)
    if protocol['corpus_hash']!=ranker.corpus.hash or protocol['config_hash']!=ranker.policy_hash:
        st.warning('Select the snapshot and policy used for this protocol.');return
    if not participant_mode:
        download('Download study protocol',protocol,'study-protocol.json')
        download('Export recorded observations',study.records(protocol),'study-observations.json')
    consent=st.checkbox('I agree to this local session recording my pseudonym, answers, clarity ratings and task duration')
    participant=st.text_input('Participant pseudonym')
    slot=st.number_input('Participant slot assigned by facilitator',min_value=0,step=1)
    if not consent or not participant.strip():return
    records=study.records(protocol,participant)
    if records and any(r['slot']!=int(slot) for r in records):
        st.warning(f"This participant is assigned to slot {records[0]['slot']}. Restore that slot to continue.");return
    complete={r['task'] for r in records if r['responses'] is not None}
    trials=assignment(protocol,int(slot));remaining=[r for r in trials if r['task']['id'] not in complete]
    if not remaining:st.success('Session complete. Thank you.');return
    trial=remaining[0];task=trial['task'];condition=trial['condition']
    st.caption(f'Task {len(complete)+1} of {len(trials)}')
    active=any(r['task']==task['id'] for r in records)
    if not active:
        if st.button('Start this task',type='primary'):
            study.start(protocol,participant,int(slot),task['id'],condition);st.rerun()
        return
    st.subheader(task['query']);system='linear' if condition.startswith('linear') else 'hybrid'
    for i,trace in enumerate(task['results'][system]):
        row={**trace,'rank':i+1,'system':system,'score':trace['scores'][system]}
        render_card(row,ranker,catalog,language,show_explanation=condition.endswith('explained'),key_prefix='study-'+task['id'])
    with st.form('study-'+task['id']):
        answers=[]
        for question in task['questions']:
            answer=st.radio(question['prompt'],range(len(question['options'])),index=None,format_func=lambda i,q=question:q['options'][i],key=task['id']+'-'+question['id'])
            answers.append(answer)
        clarity=st.selectbox('How clear is the ranking? (1 = very unclear, 5 = very clear)',options=[1,2,3,4,5],index=None)
        if st.form_submit_button('Submit task answers'):
            study.complete(protocol,participant,task['id'],answers,clarity);st.rerun()


def main():
    st.sidebar.title('OGD workbench')
    st.sidebar.caption('Transparent dataset ranking')
    participant_mode=os.environ.get('OGD_STUDY_MODE')=='participant'
    page=st.sidebar.radio('Workspace',['Study'] if participant_mode else ['Search','Collection','Analytics','Assessments','Evaluation','Calibration','Study'])
    snapshots={'Example · normalized metadata labels':ROOT/'data/legacy/corpus_v2.json',
               'Historical v1 · exact ranking inputs':ROOT/'data/legacy/corpus.json'}
    for path in sorted((LOCAL/'snapshots').glob('*/corpus.json')):snapshots[path.parent.name]=path
    chosen=st.sidebar.selectbox('Dataset snapshot',list(snapshots));path=snapshots[chosen]
    language=st.sidebar.selectbox('Display language',['en','de','fr','it'])
    default_policy=load_config(ROOT/'configs/default.json')
    manifest_path=path.with_name('manifest.json')
    if manifest_path.exists():default_policy['reference_date']=read_json(manifest_path)['completed_at']
    config=deepcopy(st.session_state.get('policy',default_policy))
    with st.sidebar.expander('Ranking preferences'):
        st.caption('Weights affect the linear score and its hybrid component. They do not rewrite fuzzy rules.')
        for feature in FEATURES:config['feature_weights'][feature]=st.slider(feature.title(),0.,2.,float(config['feature_weights'][feature]),.1,key='weight-'+feature)
        config['alpha']=st.slider('Fuzzy share in hybrid',0.,1.,float(config['alpha']),.05)
        reference=st.text_input('Reference date (ISO UTC)',value=config['reference_date']);config['reference_date']=reference
        st.caption('For a new live snapshot, choose an appropriate fixed reference date. Changing it changes the policy hash.')
        uploaded=st.file_uploader('Load policy JSON',type=['json'])
        if uploaded and st.button('Load policy'):
            proposal=validate(parse_json(uploaded.getvalue()));st.session_state.policy=proposal
            for feature in FEATURES:st.session_state.pop('weight-'+feature,None)
            st.rerun()
    validate(config)
    ranker=engine(str(path),digest(read_json(path)),json.dumps(config,sort_keys=True))
    catalog_path=path.with_name('catalog.json');catalog=Catalog(ranker.corpus,read_json(catalog_path) if catalog_path.exists() else None)
    st.sidebar.checkbox('Enable semantic model',key='semantic_enabled',help='Requires semantic extras and a one-time download of the pinned model. Runs locally on CPU.')
    filters={}
    if page=='Search':
        with st.sidebar.expander('Filter datasets',expanded=True):
            for key,values in catalog.facets().items():filters[key]=st.multiselect(key.title(),list(values),format_func=lambda x:x)
    st.sidebar.caption(f"Corpus {ranker.corpus.hash[:12]} · policy {ranker.policy_hash[:12]}")
    if page=='Search':search_page(ranker,catalog,path,language,filters)
    elif page=='Collection':collection_page()
    elif page=='Analytics':analytics_page(ranker,catalog)
    elif page=='Assessments':assessment_page(ranker,catalog,path,language)
    elif page=='Evaluation':evaluation_page(ranker,path)
    elif page=='Calibration':calibration_page(ranker)
    elif page=='Study':study_page(ranker,catalog,language)
    st.divider();st.caption('Created and maintained by Janick Spycher · Historical metadata and judgments: Deep Shukla’s thesis project; original publishers retain their data terms.')


try:
    main()
except (ValueError,KeyError,TypeError,RuntimeError,OSError) as exc:
    st.error(str(exc))
