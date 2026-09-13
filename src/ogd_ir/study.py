"""A local, counterbalanced explanation-comprehension study runner.

Software records observations; it does not supply participants or study findings.
Use a reviewed protocol and recruitment process before collecting real responses.
"""
from datetime import datetime, timezone
import json
import random
from .io import digest, canonical_bytes

CONDITIONS=('hybrid_hidden','hybrid_explained','linear_explained')


def make_protocol(ranker,queries,seed=2026):
    if len(queries)<3 or len({q['id'] for q in queries})!=len(queries):raise ValueError('Use at least three distinct study queries')
    tasks=[]
    for q in queries:
        traces=ranker.rank(q['text'],limit=len(ranker.corpus.datasets))
        if len(traces)<2:raise ValueError(f"Study query {q['id']} needs two lexical results")
        by_system={system:sorted(traces,key=lambda t:(-t['scores'][system],t['dataset_id']))[:2] for system in ('hybrid','linear')}
        tasks.append({'id':q['id'],'query':q['text'],'results':by_system,
            'questions':[
                {'id':'date','prompt':'What does the freshness feature measure?',
                 'options':['Age of the metadata modification timestamp','Age of the observations inside the dataset','Verified update frequency'],
                 'answer':0},
                {'id':'availability','prompt':'Does a larger resource count establish that the download links work?',
                 'options':['Yes','No; resource count does not verify downloads'], 'answer':1},
                {'id':'rule','prompt':'What determines the Mamdani part of the score?',
                 'options':['Only the number of matching words','Activated rules over the four normalized features','A learned prediction of assessor relevance'], 'answer':1}]})
    return {'schema_version':1,'seed':seed,'conditions':list(CONDITIONS),'tasks':tasks,
            'corpus_hash':ranker.corpus.hash,'config_hash':ranker.policy_hash,'config':ranker.config,
            'design':'Each participant sees each query once. Query order is pseudorandom; a participant slot rotates three conditions. Assign slots sequentially to balance conditions.',
            'outcomes':['objective accuracy per question','elapsed seconds','self-reported clarity 1-5'],
            'scope':'Hybrid shown/hidden compares explanations with fixed ranking. Linear explained is a simpler-system comparison, not an isolated fuzzy-rule effect.'}


def assignment(protocol,slot):
    if type(slot) is not int or slot<0:raise ValueError('Participant slot must be a nonnegative integer')
    tasks=list(protocol['tasks']);random.Random(protocol['seed']+slot//3).shuffle(tasks)
    return [{'task':task,'condition':CONDITIONS[(i+slot)%3]} for i,task in enumerate(tasks)]


class StudySession:
    def __init__(self,store):
        self.store=store
        with store.connect() as db:db.execute('''CREATE TABLE IF NOT EXISTS trials(
            protocol TEXT, participant TEXT, slot INTEGER, task TEXT, condition TEXT,
            started TEXT, responses TEXT, PRIMARY KEY(protocol,participant,task))''')

    def start(self,protocol,participant,slot,task,condition):
        if not participant.strip():raise ValueError('Enter a participant pseudonym')
        expected={r['task']['id']:r['condition'] for r in assignment(protocol,slot)}
        if expected.get(task)!=condition:raise ValueError('Condition differs from assigned protocol')
        ph=digest(protocol)
        with self.store.connect() as db:
            prior=db.execute('SELECT slot FROM trials WHERE protocol=? AND participant=? LIMIT 1',(ph,participant)).fetchone()
            if prior and prior['slot']!=slot:raise ValueError('Participant slot cannot change after a trial starts')
            occupied=db.execute('SELECT participant FROM trials WHERE protocol=? AND slot=? LIMIT 1',(ph,slot)).fetchone()
            if occupied and occupied['participant']!=participant:raise ValueError('This slot already belongs to another participant')
            db.execute('INSERT OR IGNORE INTO trials VALUES(?,?,?,?,?,?,NULL)',
                       (ph,participant,slot,task,condition,datetime.now(timezone.utc).isoformat()))

    def complete(self,protocol,participant,task,answers,clarity):
        spec=next((t for t in protocol['tasks'] if t['id']==task),None)
        if not spec or len(answers)!=len(spec['questions']):raise ValueError('Answer every question')
        if type(clarity) is not int or not 1<=clarity<=5:raise ValueError('Choose a clarity rating from 1 to 5')
        for answer,q in zip(answers,spec['questions']):
            if type(answer) is not int or not 0<=answer<len(q['options']):raise ValueError('Answer every question')
        with self.store.connect() as db:
            row=db.execute('SELECT * FROM trials WHERE protocol=? AND participant=? AND task=?',(digest(protocol),participant,task)).fetchone()
            if not row:raise ValueError('Start the trial first')
            if row['responses'] is not None:raise ValueError('Completed trial cannot be overwritten')
            elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(row['started'])).total_seconds()
            response={'answers':answers,'correct':[a==q['answer'] for a,q in zip(answers,spec['questions'])],
                      'clarity':clarity,'elapsed_seconds':elapsed,'completed_at':datetime.now(timezone.utc).isoformat()}
            db.execute('UPDATE trials SET responses=? WHERE protocol=? AND participant=? AND task=?',
                       (canonical_bytes(response).decode(),digest(protocol),participant,task))

    def records(self,protocol,participant=None):
        with self.store.connect() as db:
            sql='SELECT * FROM trials WHERE protocol=?';args=[digest(protocol)]
            if participant is not None:sql+=' AND participant=?';args.append(participant)
            return [{**dict(r),'responses':json.loads(r['responses']) if r['responses'] else None} for r in db.execute(sql,args)]
