"""Local SQLite assessments, adjudications and opt-in feedback with an audit trail."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from .io import canonical_bytes as canonical, digest


class StudyStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS pools(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS ratings(pool TEXT, query TEXT, dataset TEXT, assessor TEXT,
                grade INTEGER CHECK(grade BETWEEN 0 AND 2), note TEXT, updated TEXT,
                PRIMARY KEY(pool,query,dataset,assessor));
            CREATE TABLE IF NOT EXISTS adjudications(pool TEXT, query TEXT, dataset TEXT,
                grade INTEGER CHECK(grade BETWEEN 0 AND 2), assessor TEXT, note TEXT, basis TEXT,
                PRIMARY KEY(pool,query,dataset));
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, kind TEXT, payload TEXT, time TEXT);
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=20);db.row_factory=sqlite3.Row
        try:
            with db:yield db
        finally:db.close()

    def _event(self,db,kind,payload):
        db.execute('INSERT INTO events(kind,payload,time) VALUES(?,?,?)',
                   (kind,canonical(payload).decode(),datetime.now(timezone.utc).isoformat()))

    def add_pool(self,pool):
        queries=pool.get('queries',[])
        if not queries or len({q['id'] for q in queries})!=len(queries):raise ValueError('Invalid annotation pool')
        if any(not q['text'].strip() or not q['judgments'] or any(v is not None for v in q['judgments'].values()) for q in queries):
            raise ValueError('Start independent assessments with a nonempty, unassessed pool (null grades)')
        identity=digest(pool)
        with self.connect() as db:db.execute('INSERT OR IGNORE INTO pools VALUES(?,?)',(identity,canonical(pool).decode()))
        return identity

    def pools(self):
        with self.connect() as db:return [dict(row) for row in db.execute('SELECT id,payload FROM pools ORDER BY rowid DESC')]

    def pool(self,identity):
        with self.connect() as db:row=db.execute('SELECT payload FROM pools WHERE id=?',(identity,)).fetchone()
        if not row:raise ValueError('Unknown pool')
        return json.loads(row['payload'])

    def _check(self,pool,query,dataset,grade,assessor):
        payload=self.pool(pool)
        found=next((q for q in payload['queries'] if q['id']==query),None)
        if not found or dataset not in found['judgments']:raise ValueError('Unknown assessment item')
        if type(grade) is not int or grade not in (0,1,2):raise ValueError('Choose 0, 1 or 2; blank is unassessed')
        if not assessor.strip():raise ValueError('Enter an assessor pseudonym')

    def rate(self,pool,query,dataset,assessor,grade,note=''):
        assessor=assessor.strip();self._check(pool,query,dataset,grade,assessor)
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO ratings VALUES(?,?,?,?,?,?,?)',
                       (pool,query,dataset,assessor,grade,note,datetime.now(timezone.utc).isoformat()))
            # A changed assessment invalidates any earlier adjudication of this item.
            db.execute('DELETE FROM adjudications WHERE pool=? AND query=? AND dataset=?',(pool,query,dataset))
            self._event(db,'rating',dict(pool=pool,query=query,dataset=dataset,assessor=assessor,grade=grade,note=note))

    def ratings(self,pool,assessor=None):
        with self.connect() as db:
            sql='SELECT * FROM ratings WHERE pool=?';args=[pool]
            if assessor is not None:sql+=' AND assessor=?';args.append(assessor)
            return [dict(r) for r in db.execute(sql,args)]

    def adjudicate(self,pool,query,dataset,assessor,grade,note):
        self._check(pool,query,dataset,grade,assessor)
        if not note.strip():raise ValueError('Record an adjudication rationale')
        with self.connect() as db:
            basis=[dict(r) for r in db.execute('SELECT * FROM ratings WHERE pool=? AND query=? AND dataset=? ORDER BY assessor',(pool,query,dataset))]
            if len(basis)<2:raise ValueError('Obtain at least two independent ratings first')
            db.execute('INSERT OR REPLACE INTO adjudications VALUES(?,?,?,?,?,?,?)',
                       (pool,query,dataset,grade,assessor.strip(),note,digest(basis)))
            self._event(db,'adjudication',dict(pool=pool,query=query,dataset=dataset,assessor=assessor,grade=grade,note=note,basis=basis))

    def agreement(self,pool,assessor_a,assessor_b):
        if not assessor_a or not assessor_b or assessor_a==assessor_b:raise ValueError('Choose two different assessors')
        a={(r['query'],r['dataset']):r['grade'] for r in self.ratings(pool,assessor_a)}
        b={(r['query'],r['dataset']):r['grade'] for r in self.ratings(pool,assessor_b)}
        keys=sorted(set(a)&set(b));matrix=[[0]*3 for _ in range(3)]
        for key in keys:matrix[a[key]][b[key]]+=1
        n=len(keys)
        if not n:return {'n':0,'exact_agreement':None,'quadratic_weighted_kappa':None,'matrix':matrix}
        observed=sum(matrix[i][j]*(i-j)**2/4 for i in range(3) for j in range(3))/n
        expected=sum(sum(matrix[i])*sum(row[j] for row in matrix)/n**2*(i-j)**2/4 for i in range(3) for j in range(3))
        return {'n':n,'exact_agreement':sum(matrix[i][i] for i in range(3))/n,
                'quadratic_weighted_kappa':1-observed/expected if expected else None,'matrix':matrix,
                'note':'Overlap only; kappa undefined when expected disagreement is zero.'}

    def export(self,pool,*,attested_by):
        if not attested_by.strip():raise ValueError('A researcher must attest to the assessment process')
        payload=self.pool(pool);ratings=self.ratings(pool);queries=[]
        with self.connect() as db:
            adjudications=[dict(r) for r in db.execute('SELECT * FROM adjudications WHERE pool=?',(pool,))]
        for query in payload['queries']:
            judgments={}
            for identity in query['judgments']:
                rows=[r for r in ratings if r['query']==query['id'] and r['dataset']==identity]
                if len(rows)<2:raise ValueError(f"Incomplete independent ratings: {query['id']} / {identity}")
                grades={r['grade'] for r in rows}
                adjud=next((r for r in adjudications if r['query']==query['id'] and r['dataset']==identity),None)
                if len(grades)>1 and not adjud:raise ValueError(f"Resolve disagreement: {query['id']} / {identity}")
                judgments[identity]=adjud['grade'] if adjud else rows[0]['grade']
            queries.append({'id':query['id'],'text':query['text'],'judgments':judgments})
        return {'status':'verified','provenance':{'kind':'independent_assessment_workflow','pool_hash':pool,
                'pool_provenance':payload['provenance'],'attested_by':attested_by,'ratings':ratings,
                'adjudications':adjudications,'caveat':'Pseudonyms are not identity authentication; independence is researcher-attested.'},'queries':queries}

    def feedback(self,payload):
        if payload.get('rating') not in ('useful','not_useful'):raise ValueError('Invalid feedback')
        with self.connect() as db:self._event(db,'feedback',payload)

    def feedback_rows(self):
        with self.connect() as db:return [dict(r) for r in db.execute("SELECT * FROM events WHERE kind='feedback' ORDER BY id")]
