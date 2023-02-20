"""
Functions required by 002_Single Document.py
"""
import numpy as np
import os
import pandas as pd
from dpr import core

def check_if_encoded(file_name):
    file_name = file_name+".npy"
    if file_name in os.listdir("single_doc/"):
        return True

def load_encoded(file_name):
    (doc_emb,spans,file_names,text) = np.load('single_doc/'+file_name,allow_pickle='TRUE')

    doc_emb = np.array(list(doc_emb.values())).reshape(-1,768)
    spans = list(spans.values())
    file_names = list(file_names.values())
    
    return doc_emb,spans,file_names,text

def save_encoded(doc_emb, spans, file_names,file_name,text):
    emb = dict(zip(list(range(len(doc_emb))),doc_emb))
    spans = dict(zip(list(range(len(spans))),spans))
    file_names = dict(zip(list(range(len(file_names))),file_names))

    np.save("single_doc/"+file_name,(emb,spans,file_names,text)) 
    
def predict(query,text,file_name):
    if check_if_encoded(file_name):
        doc_emb,spans,file_names,text = load_encoded(file_name+".npy")
    else:
        doc_emb, spans, file_names,text = core.encode_docs((file_name,text),maxlen = 128, stride = 64)
        save_encoded(doc_emb, spans, file_names,file_name+".npy",text)
    
    query_emb = core.encode_query(query)
    df = core.create_output(query,query_emb,doc_emb,spans, file_names)
    
    return df

def get_answer(query,data,file_name):
    text = core.extract_data(data)

    path,flag = core.check_log_history(query,file_name,text)
    
    if not flag:
         df = predict(query,text,file_name)
         df.to_csv("HISTORY/{}".format(path),index=False)
    else:
        try:
            #in case the read fails
            df = pd.read_csv("HISTORY/{}".format(path))
        except:
            #just act as if it's new query
            #st.write(e)
            df = predict(query,text,file_name)
            df.to_csv("HISTORY/{}".format(path),index=False)
    return df
