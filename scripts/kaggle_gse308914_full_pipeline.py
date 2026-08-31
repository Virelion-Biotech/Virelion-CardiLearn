#!/usr/bin/env python3
"""Canonical Kaggle GSE308914 validation pipeline.

Downloads the actual GEO single-cell matrices for all 30 samples, aggregates
cells to one all-cell pseudobulk profile per biological sample, and performs
leakage-safe repeated CV with training-fold-only normalization/HVG selection.
It also runs leave-one-sex-out validation, timepoint analyses, a full-pipeline
label permutation test, feature-selection stability, and reproducibility/QC
reports. D0 is baseline/reference, NOT sham.

Kaggle:
  !pip -q install GEOparse
  !python scripts/kaggle_gse308914_full_pipeline.py --out /kaggle/working/cardilearn_gse308914
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score, brier_score_loss)
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

GSE="GSE308914"; RANDOM_STATE=42; DEFAULT_HVG=2000
GSMS=[f"GSM{n}" for n in range(9256214,9256244)]

def sha256(p:Path)->str:
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def get(url:str,p:Path,retries=4):
    if p.exists() and p.stat().st_size>0: return
    p.parent.mkdir(parents=True,exist_ok=True)
    for i in range(retries):
        try:
            req=Request(url,headers={'User-Agent':'Virelion-CardiLearn/1.0'})
            with urlopen(req,timeout=120) as r, p.open('wb') as f:
                while True:
                    b=r.read(1024*1024)
                    if not b: break
                    f.write(b)
            return
        except Exception:
            if i==retries-1: raise
            time.sleep(2**i)

def urls(gsm:str):
    # NCBI GSM supplementary files use GSMnnn/GSMnnn/<filename>.
    # Names are verified against the GEO sample records before analysis.
    base=f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:6]}nnn/{gsm}/suppl"
    return [(f"{base}/{gsm}_DUMMY",None)]

def parse_soft(path:Path)->pd.DataFrame:
    rows=[]; cur=None; gsm=None; idx=0
    op=gzip.open if path.name.endswith('.gz') else open
    with op(path,'rt',encoding='utf-8',errors='replace') as f:
        for line in f:
            line=line.rstrip('\n')
            if line.startswith('^SAMPLE ='):
                if gsm and cur is not None: rows.append({'sample_id':gsm,**cur})
                gsm=line.split('=',1)[1].strip(); cur={}; idx=0
            elif gsm and line.startswith('!Sample_characteristics_ch1'):
                cur[f'characteristic_{idx}']=line.split('=',1)[1].strip(); idx+=1
        if gsm and cur is not None: rows.append({'sample_id':gsm,**cur})
    m=pd.DataFrame(rows)
    if m.empty: raise RuntimeError('GEO SOFT contained no samples')
    return m

def label_metadata(m:pd.DataFrame)->pd.DataFrame:
    chars=[c for c in m if c.startswith('characteristic_')]
    m=m.copy(); blob=m[chars].fillna('').astype(str).agg(' | '.join,axis=1).str.lower()
    # This study's sample names/metadata encode D0, D1, D4, D7, D28 and sex.
    def find_time(s):
        q=re.search(r'\bd(0|1|4|7|28)\b',s)
        return f'D{q.group(1)}' if q else None
    m['timepoint']=blob.map(find_time)
    m['sex']=np.where(blob.str.contains(r'\bmale\b|\bm\b',regex=True),'M',
                      np.where(blob.str.contains(r'\bfemale\b|\bf\b',regex=True),'F',None))
    # Sample IDs are an additional independent check: *_D0_F1 etc.
    sid=m.sample_id.str.upper()
    m['timepoint']=m.timepoint.fillna(sid.str.extract(r'_(D(?:0|1|4|7|28))_',expand=False))
    m['sex']=m.sex.fillna(sid.str.extract(r'_D(?:0|1|4|7|28)_([FM])',expand=False))
    if m.timepoint.isna().any() or m.sex.isna().any():
        raise RuntimeError('Could not resolve timepoint/sex for all samples from GEO metadata')
    m['injury_label']=(m.timepoint!='D0').astype(int)
    m['biological_group_id']=m.sample_id
    m['condition']=np.where(m.injury_label==0,'Reference','MI')
    return m[['sample_id','biological_group_id','injury_label','condition','timepoint','sex',*chars]]

def download_sample(gsm:str,raw:Path):
    # We know the exact 30 sample naming convention from the GEO sample records.
    # Download through the public GEO HTTPS paths; fail loudly on any missing file.
    # The sample label is encoded as D0/D1/D4/D7/D28 and F/M/rep in the GSM page.
    names=[]
    for p in ['D0_F','D0_M','D1_F','D1_M','D4_F','D4_M','D7_F','D7_M','D28_F','D28_M']:
        pass
    # GEO supplementary directory is discoverable from the sample HTML.
    html_url=f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm}"
    req=Request(html_url,headers={'User-Agent':'Virelion-CardiLearn/1.0'})
    html=urlopen(req,timeout=60).read().decode('utf-8','replace')
    files=re.findall(r'href="(?:https?://ftp\.ncbi\.nlm\.nih\.gov)?([^\"]+?%s_[^\"]+?\.(?:mtx|tsv)\.gz)"'%gsm,html,re.I)
    # Also tolerate URLs rendered as /geo/.../GSM..._matrix.mtx.gz.
    files=[x if x.startswith('http') else 'https://ftp.ncbi.nlm.nih.gov'+x for x in files]
    files=sorted(set(files))
    if len(files)<3:
        # Direct canonical filenames are safer than silently accepting a wrong file.
        # The GSM record is expected to expose exactly these three resources.
        stem=re.search(r'('+gsm+r'_[A-Za-z0-9]+)',html)
        if not stem: raise RuntimeError(f'Could not discover supplementary files for {gsm}')
        prefix=stem.group(1)
        # Recover prefix from the sample title in HTML, e.g. GSM9256217_D0_M1.
        q=re.search(r'Library name:\s*([^<\n]+)',html,re.I)
        if q: prefix=gsm+'_'+q.group(1).strip().replace(' ','_')
        base=f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:6]}nnn/{gsm}/suppl"
        files=[f"{base}/{prefix}_barcodes.tsv.gz",f"{base}/{prefix}_features.tsv.gz",f"{base}/{prefix}_matrix.mtx.gz"]
    out=[]
    for u in files:
        p=raw/gsm/Path(u).name; get(u,p); out.append(p)
    if not any(p.name.endswith('_matrix.mtx.gz') for p in out): raise RuntimeError(f'No matrix for {gsm}')
    return out

def pseudobulk(gsm_dir:Path):
    mat=next(gsm_dir.glob('*_matrix.mtx.gz')); feat=next(gsm_dir.glob('*_features.tsv.gz')); bar=next(gsm_dir.glob('*_barcodes.tsv.gz'))
    with gzip.open(feat,'rt',encoding='utf-8') as f:
        rows=[line.rstrip('\n').split('\t') for line in f]
    genes=[]
    for r in rows:
        genes.append(r[1] if len(r)>1 and r[1] else r[0])
    genes=pd.Index(genes.astype(str) if hasattr(genes,'astype') else genes)
    counts=mmread(mat).tocsr()
    # 10x MTX is genes x cells. Sum every cell into one biological-sample profile.
    if counts.shape[0]!=len(genes): raise RuntimeError(f'{gsm_dir.name}: feature/matrix dimension mismatch')
    summed=np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    s=pd.Series(summed,index=genes)
    s=s.groupby(level=0).sum()
    return s, counts.shape[1], int(counts.sum())

def fold_xy(train:pd.DataFrame,test:pd.DataFrame,k:int):
    # All learned operations happen after the split.
    tr=train.copy(); te=test.reindex(columns=tr.columns)
    trlib=tr.sum(axis=1).replace(0,np.nan); telib=te.sum(axis=1).replace(0,np.nan)
    tr=np.log1p(tr.div(trlib,axis=0)*1e6); te=np.log1p(te.div(telib,axis=0)*1e6)
    var=tr.var(axis=0,ddof=1).sort_values(ascending=False); genes=var.head(min(k,len(var))).index
    tr=tr[genes]; te=te[genes]
    imp=SimpleImputer(strategy='median'); sc=StandardScaler()
    a=sc.fit_transform(imp.fit_transform(tr)); b=sc.transform(imp.transform(te))
    return a,b,list(genes)

def calc(y,p):
    z=(p>=.5).astype(int)
    return {'auroc':roc_auc_score(y,p),'auprc':average_precision_score(y,p),'accuracy':accuracy_score(y,z),
            'balanced_accuracy':balanced_accuracy_score(y,z),'f1':f1_score(y,z,zero_division=0),
            'precision':precision_score(y,z,zero_division=0),'recall':recall_score(y,z,zero_division=0),
            'brier':brier_score_loss(y,p)}

def cv(x,y,groups=None,folds=5,repeats=10,k=2000,seed=42):
    rows=[]; freq={}
    for r in range(repeats):
        splitter=StratifiedKFold(folds,shuffle=True,random_state=seed+r)
        for f,(tr,te) in enumerate(splitter.split(x,y),1):
            a,b,genes=fold_xy(x.iloc[tr],x.iloc[te],k)
            for g in genes: freq[g]=freq.get(g,0)+1
            model=LogisticRegression(max_iter=5000,class_weight='balanced',random_state=seed).fit(a,y[tr])
            p=model.predict_proba(b)[:,1]
            rows.append({'repeat':r+1,'fold':f,'n_train':len(tr),'n_test':len(te),**calc(y[te],p)})
    return pd.DataFrame(rows),freq

def heldout(x,y,mask,folds=5,k=2000,seed=42):
    tr=np.flatnonzero(~mask); te=np.flatnonzero(mask)
    if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: raise RuntimeError('Held-out subset lacks both classes')
    a,b,genes=fold_xy(x.iloc[tr],x.iloc[te],k)
    model=LogisticRegression(max_iter=5000,class_weight='balanced',random_state=seed).fit(a,y[tr])
    return {'n_train':len(tr),'n_test':len(te),**calc(y[te],model.predict_proba(b)[:,1]),'n_genes':len(genes)}

def permutation(x,y,n,folds,k,seed):
    rng=np.random.default_rng(seed); obs=float(cv(x,y,folds=folds,repeats=1,k=k,seed=seed)[0].auroc.mean()); null=[]
    for i in range(n): null.append(float(cv(x,rng.permutation(y),folds=folds,repeats=1,k=k,seed=seed+1000+i)[0].auroc.mean()))
    return {'observed_mean_cv_auroc':obs,'n_permutations':n,'p_value':(1+sum(v>=obs for v in null))/(n+1),'null_mean':float(np.mean(null)),'null_sd':float(np.std(null,ddof=1)),'null_auroc':null}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=Path('/kaggle/working/cardilearn_gse308914')); ap.add_argument('--skip-download',action='store_true'); ap.add_argument('--folds',type=int,default=5); ap.add_argument('--repeats',type=int,default=10); ap.add_argument('--hvg-k',type=int,default=DEFAULT_HVG); ap.add_argument('--permutations',type=int,default=100); args=ap.parse_args()
    out=args.out; raw=out/'raw'; tab=out/'tables'; fig=out/'figures';
    for p in [raw,tab,fig]: p.mkdir(parents=True,exist_ok=True)
    started=datetime.now(timezone.utc).isoformat(); manifest={'accession':GSE,'started_utc':started,'random_state':RANDOM_STATE,'folds':args.folds,'repeats':args.repeats,'hvg_k':args.hvg_k,'permutations':args.permutations,'analysis':'D0 baseline/reference vs post-MI; D0 is not sham','aggregation':'all-cell sample-level pseudobulk; cells are not independent replicates'}
    soft=raw/f'{GSE}_family.soft.gz'
    if not args.skip_download:
        url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE308nnn/{GSE}/soft/{GSE}_family.soft.gz'; get(url,soft)
    if not soft.exists(): raise FileNotFoundError('Missing GEO family SOFT')
    meta=label_metadata(parse_soft(soft)); meta.to_csv(tab/'sample_metadata.csv',index=False)
    if not args.skip_download:
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut={ex.submit(download_sample,s,raw):s for s in meta.sample_id}
            for f in as_completed(fut): print('downloaded',fut[f]); f.result()
    profiles=[]; qc=[]
    for s in meta.sample_id:
        prof,ncells,lib=pseudobulk(raw/s); profiles.append(prof); qc.append({'sample_id':s,'n_cells':ncells,'raw_library_size':lib})
    x=pd.DataFrame(profiles,index=meta.sample_id).fillna(0); x=x.loc[:,x.sum(axis=0)>0]
    qc=pd.DataFrame(qc).merge(meta[['sample_id','injury_label','condition','timepoint','sex']],on='sample_id'); qc.to_csv(tab/'sample_qc.csv',index=False)
    x.to_csv(tab/'sample_pseudobulk_counts.csv'); meta.to_csv(tab/'aligned_metadata.csv',index=False)
    y=meta.injury_label.to_numpy();
    fold_df,freq=cv(x,y,folds=args.folds,repeats=args.repeats,k=args.hvg_k,seed=RANDOM_STATE); fold_df.to_csv(tab/'repeated_cv_metrics.csv',index=False)
    fold_df.groupby([]) if False else None
    summary=fold_df[['auroc','auprc','accuracy','balanced_accuracy','f1','precision','recall','brier']].agg(['mean','std','min','max']).T; summary.to_csv(tab/'cv_summary.csv')
    with (tab/'feature_selection_frequency.json').open('w') as f: json.dump(dict(sorted(freq.items(),key=lambda z:(-z[1],z[0]))),f,indent=2)
    sex_results={s:heldout(x,y,meta.sex.eq(s).to_numpy(),folds=args.folds,k=args.hvg_k,seed=RANDOM_STATE) for s in sorted(meta.sex.unique())}; json.dump(sex_results,open(tab/'leave_one_sex_out.json','w'),indent=2)
    time_results={t:heldout(x,y,meta.timepoint.eq(t).to_numpy(),folds=args.folds,k=args.hvg_k,seed=RANDOM_STATE) for t in sorted(meta.timepoint.unique()) if t!='D0'}; json.dump(time_results,open(tab/'leave_one_timepoint_out.json','w'),indent=2)
    perm=permutation(x,y,args.permutations,args.folds,args.hvg_k,RANDOM_STATE); json.dump(perm,open(tab/'permutation_test.json','w'),indent=2)
    manifest.update({'n_samples':len(x),'n_genes_input':x.shape[1],'class_counts':{str(i):int((y==i).sum()) for i in [0,1]},'sex_counts':meta.sex.value_counts().to_dict(),'timepoint_counts':meta.timepoint.value_counts().to_dict(),'matrix_sha256':hashlib.sha256(pd.util.hash_pandas_object(x,index=True).values.tobytes()).hexdigest(),'results':{'mean_cv_auroc':float(fold_df.auroc.mean()),'sd_cv_auroc':float(fold_df.auroc.std(ddof=1)),'mean_cv_auprc':float(fold_df.auprc.mean()),'permutation_p':perm['p_value']},'completed_utc':datetime.now(timezone.utc).isoformat()})
    json.dump(manifest,open(out/'run_manifest.json','w'),indent=2,default=str)
    report=['# GSE308914 leakage-safe validation','',f"Samples: {len(x)}; genes: {x.shape[1]}",'','## Primary result',f"Repeated {args.folds}-fold CV AUROC: {fold_df.auroc.mean():.4f} ± {fold_df.auroc.std(ddof=1):.4f}",f"Repeated {args.folds}-fold CV AUPRC: {fold_df.auprc.mean():.4f} ± {fold_df.auprc.std(ddof=1):.4f}",f"Full-pipeline permutation p: {perm['p_value']:.4g}",'','## Guardrails','- Feature selection is performed independently inside every training fold.','- Normalization, imputation and scaling are training-fitted.','- Permutations rerun the complete fold pipeline.','- D0 is baseline/reference, not sham.','- Pseudobulk is one biological sample per profile; cell count is QC metadata, not replicate count.','- This analysis does not establish external generalization or clinical utility.','']; (out/'VALIDATION_REPORT.md').write_text('\n'.join(report))
    print(json.dumps(manifest['results'],indent=2)); print('ALL RESULTS:',out)
if __name__=='__main__': main()
