"""Actual multilingual MiniLM retrieval with a pinned CPU ONNX encoder.

Mean pooling and L2 normalization follow the model card. No lexical fallback.
The quantized ONNX artifact is declared separately from the original PyTorch model.
"""
from pathlib import Path
import hashlib
import os
import tempfile
import numpy as np
from .io import digest, read_json, write_json

MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION = 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
ARTIFACT = 'onnx/model_quint8_avx2.onnx'


class MiniLMEncoder:
    def __init__(self, cache_dir=None, offline=False):
        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('Install semantic dependencies: pip install -e ".[semantic]"') from exc
        paths={name:hf_hub_download(MODEL_ID,name,revision=REVISION,cache_dir=cache_dir,
                                    local_files_only=offline) for name in (ARTIFACT,'tokenizer.json')}
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(paths[ARTIFACT],sess_options=options,providers=['CPUExecutionProvider'])
        self.tokenizer=Tokenizer.from_file(paths['tokenizer.json'])
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id('<pad>'),pad_token='<pad>')
        self.provenance={'model':MODEL_ID,'revision':REVISION,'artifact':ARTIFACT,
            'artifact_sha256':hashlib.sha256(Path(paths[ARTIFACT]).read_bytes()).hexdigest(),
            'tokenizer_sha256':hashlib.sha256(Path(paths['tokenizer.json']).read_bytes()).hexdigest(),
            'max_tokens':128,'document_batch_size':16,'query_batch_size':1,'pooling':'attention-mask mean','normalization':'L2',
            'runtime':'onnxruntime CPUExecutionProvider','runtime_version':ort.__version__,
            'quantization':'unsigned 8-bit AVX2 export; scores need not equal float PyTorch outputs'}

    def encode(self, texts, batch_size=16):
        if not texts: return np.empty((0,384),dtype=np.float32)
        batches=[]; names={x.name for x in self.session.get_inputs()}
        for start in range(0,len(texts),batch_size):
            encoded=self.tokenizer.encode_batch(texts[start:start+batch_size])
            mask=np.array([x.attention_mask for x in encoded],dtype=np.int64)
            inputs={'input_ids':np.array([x.ids for x in encoded],dtype=np.int64),'attention_mask':mask,
                    'token_type_ids':np.array([x.type_ids for x in encoded],dtype=np.int64)}
            hidden=self.session.run(None,{k:v for k,v in inputs.items() if k in names})[0]
            pooled=(hidden*mask[:,:,None]).sum(axis=1)/np.maximum(mask.sum(axis=1,keepdims=True),1)
            batches.append(pooled/np.maximum(np.linalg.norm(pooled,axis=1,keepdims=True),1e-12))
        return np.concatenate(batches).astype(np.float32)


class SemanticIndex:
    def __init__(self, corpus, encoder, cache_dir=None):
        self.corpus=corpus;self.encoder=encoder;self.ids=list(corpus.by_id)
        self.metadata={'schema_version':1,'corpus_hash':corpus.hash,'encoder':encoder.provenance,
                       'text_policy':'title + description + keywords + themes, separated by newlines; tokenizer truncates at 128 tokens'}
        self.key=digest(self.metadata); self.embeddings=None
        cache=Path(cache_dir)/self.key if cache_dir else None
        if cache and cache.with_suffix('.json').exists() and cache.with_suffix('.npy').exists():
            manifest=read_json(cache.with_suffix('.json'));payload=cache.with_suffix('.npy').read_bytes()
            if manifest.get('metadata')==self.metadata and hashlib.sha256(payload).hexdigest()==manifest.get('sha256'):
                self.embeddings=np.load(cache.with_suffix('.npy'),allow_pickle=False)
        if self.embeddings is None:
            texts=['\n'.join([d.title,d.description,d.keywords,d.themes]) for d in corpus.datasets]
            self.embeddings=encoder.encode(texts)
            self._validate()
            if cache:
                cache.parent.mkdir(parents=True,exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=cache.parent,delete=False) as f:
                    temp=Path(f.name);np.save(f,self.embeddings,allow_pickle=False)
                os.replace(temp,cache.with_suffix('.npy'))
                write_json(cache.with_suffix('.json'),{'metadata':self.metadata,'sha256':hashlib.sha256(cache.with_suffix('.npy').read_bytes()).hexdigest()})
        self._validate()

    def _validate(self):
        if self.embeddings.ndim!=2 or self.embeddings.shape[0]!=len(self.ids) or not np.isfinite(self.embeddings).all():
            raise ValueError('Invalid semantic embedding cache')
        if not np.allclose(np.linalg.norm(self.embeddings,axis=1),1,atol=1e-4):raise ValueError('Embeddings must have unit norm')

    def scores(self, query):
        if not query.strip():raise ValueError('Empty semantic query')
        vector=self.encoder.encode([query])[0]
        if not np.isfinite(vector).all():raise ValueError('Nonfinite query embedding')
        scores=np.clip(self.embeddings@vector,-1,1)
        return {identity:float(score) for identity,score in zip(self.ids,scores)}

    def rank(self, query, candidates=None, limit=10):
        ids=list(candidates) if candidates is not None else self.ids
        if limit<1 or any(x not in self.corpus.by_id for x in ids):raise ValueError('Invalid candidates or limit')
        scores=self.scores(query)
        return [{'dataset_id':i,'score':scores[i],'rank':rank+1,'system':'semantic',
                 'title':self.corpus.by_id[i].title,'semantic_provenance':self.metadata,
                 'explanation':'Cosine similarity of multilingual embeddings; no token-level causal attribution is claimed.'}
                for rank,i in enumerate(sorted(set(ids),key=lambda i:(-scores[i],i))[:limit])]
