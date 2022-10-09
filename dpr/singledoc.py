"""
Functions required by 002_Single Document.py
"""
import numpy as np
import os
import pandas as pd
from dpr import core

def check_if_encoded(doc_name):
    doc_name = doc_name+".npy"
    if doc_name in os.listdir("single_doc/"):
        return True

def load_encoded(doc_name):
    (doc_emb,spans,file_names) = np.load('single_doc/'+doc_name,allow_pickle='TRUE')

    doc_emb = np.array(list(doc_emb.values())).reshape(-1,768)
    spans = list(spans.values())
    file_names = list(file_names.values())
    
    return doc_emb,spans,file_names

def save_encoded(doc_emb, spans, file_names,doc_name):
    emb = dict(zip(list(range(len(doc_emb))),doc_emb))
    spans = dict(zip(list(range(len(spans))),spans))
    file_names = dict(zip(list(range(len(file_names))),file_names))

    np.save("single_doc/"+doc_name,(emb,spans,file_names)) 
    
def predict(query,text,doc_name):
    if check_if_encoded(doc_name):
        doc_emb,spans,file_names = load_encoded(doc_name+".npy")
    else:
        doc_emb, spans, file_names = core.encode_docs((doc_name,text),maxlen = 64, stride = 32)
        save_encoded(doc_emb, spans, file_names,doc_name+".npy")
    
    query_emb = core.encode_query(query)
    df = core.create_output(query,query_emb,doc_emb,spans, file_names)
    
    return df

def get_answer(query,data,doc_name):
    text = core.extract_data(data)
    #check if the same question was already asked uing the same document
    #if yes, just recover the answer
    path,flag = core.check_log_history(query,doc_name,text)
    
    if not flag:
         df = predict(query,text,doc_name)
         df.to_csv("HISTORY/{}".format(path),index=False)
    else:
        try:
            #in case the read fails
            df = pd.read_csv("HISTORY/{}".format(path))
        except:
            #just act as if it's new query
            #st.write(e)
            df = predict(query,text,doc_name)
            df.to_csv("HISTORY/{}".format(path),index=False)
    return df
