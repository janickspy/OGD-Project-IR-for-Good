"""Explicit multilingual query assistance. Search uses the user-approved text."""
import os
import re
import unicodedata
from .model import tokens

STOPWORDS = {
 'de':set('der die das den dem des ein eine einer und oder von für mit im in zu auf schweiz bitte'.split()),
 'fr':set('le la les de du des et ou pour avec en dans suisse une un'.split()),
 'it':set('il lo la gli le di del della e o per con in svizzera una un'.split()),
 'en':set('the a an and or of for with in on switzerland please'.split())}
CONCEPTS = {
 'transport':('verkehr','mobilität','transport','mobilité','trasporti','mobility'),
 'population':('bevölkerung','population','popolazione'),
 'environment':('umwelt','environnement','ambiente','environment'),
 'energy':('energie','énergie','energia','energy'),
 'employment':('beschäftigung','emploi','occupazione','employment'),
 'accidents':('verkehrsunfälle','accidents','incidenti','unfälle')}
RECENT={'aktuell','neueste','récent','récents','recente','recenti','latest','recent'}
HISTORICAL={'historisch','historische','historical','historique','storico','storici'}


def parse_query(text, language='auto', expand=False):
    clean = ' '.join(unicodedata.normalize('NFKC',str(text)).split())
    if not clean: raise ValueError('Enter a query')
    if len(clean)>2000: raise ValueError('Query is too long (maximum 2000 characters)')
    terms = tokens(clean); counts={lang:len(set(terms)&words) for lang,words in STOPWORDS.items()}
    winners=[lang for lang,n in counts.items() if n==max(counts.values()) and n>0]
    inferred=winners[0] if len(winners)==1 else 'und'
    if language not in ('auto','de','fr','it','en'):raise ValueError('Unsupported query language')
    detected=inferred if language=='auto' else language
    concepts=[key for key,words in CONCEPTS.items() if set(words)&set(terms)]
    additions=sorted({word for key in concepts for word in CONCEPTS[key]}-set(terms)) if expand else []
    return {'original':text,'normalized':clean,'language':detected,
            'language_method':'explicit' if language!='auto' else 'stopword heuristic; ambiguous is undetermined',
            'concepts':concepts,'years':sorted(set(re.findall(r'\b(?:19|20)\d{2}\b',clean))),
            'temporal_intent':'historical' if set(terms)&HISTORICAL else 'recent' if set(terms)&RECENT else None,
            'expansion_terms':additions,'search_text':' '.join([clean,*additions]),
            'note':'Years and temporal words are retained; metadata modification dates do not establish temporal coverage.'}


def llm_suggestion(text, *, model, session=None):
    """Optional real API call. No mock result or automatic query replacement."""
    import requests
    key=os.environ.get('OPENAI_API_KEY')
    if not key:raise ValueError('Set OPENAI_API_KEY on the server to enable optional query assistance')
    if not model.strip():raise ValueError('Specify an available OpenAI model')
    clean=parse_query(text)['normalized']; session=session or requests.Session()
    response=session.post('https://api.openai.com/v1/chat/completions',headers={'Authorization':f'Bearer {key}'},
        json={'model':model,'messages':[{'role':'system','content':'Rewrite the user query for searching Swiss open government datasets. Preserve dates, places and intent. Return only a concise search query, no answers or invented dataset names.'},{'role':'user','content':clean}]},timeout=30)
    if response.status_code>=400:raise RuntimeError(f'Query assistance failed (HTTP {response.status_code})')
    result=response.json()['choices'][0]['message']['content']
    return {'original':clean,'suggestion':parse_query(result)['normalized'],'provider':'OpenAI','model':model,
            'applied':False}
