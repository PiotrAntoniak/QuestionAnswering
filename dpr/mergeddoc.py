"""
Functions required by 003_Multiple Documents.py
"""
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from dpr import core
from dpr import singledoc
from datetime import datetime
import hashlib
from dpr import fragmentation
import streamlit as st

def check_if_encoded(doc_name):
    doc_name = "emb_"+doc_name+".npy"
    if doc_name in os.listdir("single_doc"):
        return True
    return False

def merge_update_metadata(meta_key,query,doc_name,metadata):
    #note that the file was used to create new emb.py and is now encoded
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    q_a = str(hashlib.sha256(str([query,doc_name]).encode()).hexdigest())
    metadata.update({meta_key:{q_a:{"query":query,
                                    "doc_name":doc_name,
                                    "time":current_time}}})
    return metadata

def create_new_merge(use_docs_dump=True):
    #recover history of queries
    metadata = core.load_metadata()
    metadata_keys = metadata.keys()
    #iterate over folder with docs and get those that were not yet encoded
    #and encode them + save in the same directory as single docs are
    if use_docs_dump:
        st.write("please wait, processing "+str(len(os.listdir("docs_dump")))+" documents")
        my_bar = st.progress(0)
        for i,doc_name in enumerate(os.listdir("docs_dump")):
            my_bar.progress(i)
            text = core.extract_saved_text("docs_dump/"+doc_name)
            print(text[:10])
            if text is not None:
                meta_key = str(hashlib.sha256(text.encode()).hexdigest())
                if meta_key not in metadata_keys:
                    doc_emb, spans, file_names = core.encode_docs((doc_name,text),maxlen = 64, stride = 32)
                    singledoc.save_encoded(doc_emb, spans, file_names,doc_name+".npy")
                    metadata = merge_update_metadata(meta_key,"merge",doc_name,metadata)
                    core.save_text(meta_key,doc_name,text)
        
    metadata.update({"last_update":datetime.now().strftime("%d-%m-%Y %H:%M:%S")})
    core.save_metadata(metadata)
    
    embs = []
    spans = []
    files_names = []
    
    #read all encoded docs
    for doc_name in tqdm(os.listdir("single_doc")):
        doc_emb,doc_text,file_names = singledoc.load_encoded(doc_name)
        files_names.extend(file_names)
        spans.extend(doc_text)
        embs.append(doc_emb)
    #and merge them
    doc_emb = np.concatenate(embs,axis=0).reshape(-1,embs[0].shape[-1])
    np.save('merged_docs/files.npy',dict(zip(list(range(len(files_names))),files_names)))
    np.save('merged_docs/spans.npy',dict(zip(list(range(len(spans))),spans)))
    np.save('merged_docs/emb.npy',dict(zip(list(range(len(doc_emb))),doc_emb)))
    fragmentation.cluster() 

def predict(query,rerun_merge=False,use_docs_dump=True,use_kmeans = False):
    if "emb.npy" not in os.listdir("merged_docs") or rerun_merge:
        create_new_merge(use_docs_dump) 
    
    query_emb = core.encode_query(query)
    
    if use_kmeans: #if no new emb,py is created, use kmeans clusters for speed
        doc_emb,spans,files_names = fragmentation.load_cluster(query_emb)
    else: #or go standard route
        doc_emb = np.load('merged_docs/emb.npy',allow_pickle='TRUE').item()
        doc_text = np.load('merged_docs/spans.npy',allow_pickle='TRUE').item()
        file_names_dicto = np.load('merged_docs/files.npy',allow_pickle='TRUE').item()
        
        doc_emb = np.array(list(doc_emb.values())).reshape(-1,768)
        spans = list(doc_text.values())
        files_names = list(file_names_dicto.values())
            
    
    df = core.create_output(query,query_emb,doc_emb,spans, files_names)
    
    return df

def get_answer(query,doc_name,rerun_merge,use_kmeans):
    doc_name = str(hashlib.sha256(doc_name.encode()).hexdigest()) #cant really have a date as text so ...
    path,flag = core.check_log_history(query,doc_name,"merged, kmeans: "+str(use_kmeans))
    if not flag:
         df = predict(query,rerun_merge,use_kmeans=use_kmeans)
         df.to_csv("HISTORY/{}".format(path),index=False)
    else:
        try:
            df = pd.read_csv("HISTORY/{}".format(path))
        except:
            df = predict(query,rerun_merge)
            df.to_csv("HISTORY/{}".format(path),index=False)
    return df

