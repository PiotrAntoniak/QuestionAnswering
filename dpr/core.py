"""
Functions used in other modules. As the name suggests, they are core functions. 
"""
import numpy as np
import hashlib
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import tokenizers
from tqdm import tqdm
import pdfplumber
from scipy.special import softmax
import pandas as pd
import streamlit as st
import json
import os
from datetime import datetime
from pdfminer.high_level import extract_text as extract_text_pdf
import docx2txt

@st.cache(allow_output_mutation=True, 
          hash_funcs={tokenizers.Tokenizer: lambda _: None, 
          tokenizers.AddedToken: lambda _: None})
def load_models():
    #loads models
    m_emb = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    tokenizer = AutoTokenizer.from_pretrained(m_emb)
    model = AutoModel.from_pretrained(m_emb).to(device).eval()
    m_ans = "deepset/roberta-base-squad2"
    tokenizer_ans = AutoTokenizer.from_pretrained(m_ans)
    model_ans = AutoModelForQuestionAnswering.from_pretrained(m_ans).to(device).eval()
    if device == 'cuda:0':
        pipe = pipeline("question-answering",model_ans,tokenizer =tokenizer_ans,device = 0)
    else:
        pipe = pipeline("question-answering",model_ans,tokenizer =tokenizer_ans)
    
    return tokenizer, tokenizer_ans, model, model_ans, pipe

tokenizer, tokenizer_ans, model, model_ans, pipe = load_models()

def cls_pooling(model_output):
    return model_output.last_hidden_state[:,0]

def encode_query(query):
    encoded_input = tokenizer(query, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)

    embeddings = cls_pooling(model_output)

    return embeddings.cpu()

def encode_docs(docs,maxlen = 64, stride = 32):
    #splits text into overlapping spans and encodes them using transformer loaded above
    encoded_input = []
    embeddings = []
    spans = []
    file_names = []
    doc_name, text = docs
        
    text = text.split(" ")
    if len(text) < maxlen:
        text = " ".join(text)
        
        encoded_input.append(tokenizer(text, return_tensors='pt', truncation = True).to(device))
        spans.append(text)
        file_names.append(doc_name)

    else:
        num_iters = int(len(text)/maxlen)+1
        
        
        for i in range(num_iters):
            
            if i == 0:
                temp_text = " ".join(text[i*maxlen:(i+1)*maxlen+stride])
                temp_text = " ".join(temp_text.split(" ")[:-1]) #remove loose chars
            else:
                temp_text = " ".join(text[(i-1)*maxlen:(i)*maxlen][-stride:] + text[i*maxlen:(i+1)*maxlen])
                temp_text = " ".join(temp_text.split(" ")[1:-1])

            encoded_input.append(tokenizer(temp_text, return_tensors='pt', truncation = True).to(device))
            spans.append(temp_text)
            file_names.append(doc_name)

    with torch.no_grad():
        for i,encoded in tqdm(enumerate(encoded_input)): 
            model_output = model(**encoded, return_dict=True)
            embeddings.append(cls_pooling(model_output))
    
    embeddings = np.float32(torch.stack(embeddings).transpose(0, 1).cpu())
    
    return embeddings, spans, file_names

def clean_text(text):
    text = text.replace("\n","")
    text = text.replace(" . "," ")
    text = text.replace("\r", " ")
    return text
    
def extract_saved_text(path):
    if path.endswith(".pdf"):
        return extract_text_pdf(path)
    elif path.endswith(".docx"):
        return docx2txt.process(path)
    else:
        return None
          
def extract_data(feed):
    #this is for pdf documents
    data = []
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        for p in pages:
            data.append(p.extract_text())
    data = ''.join(data)
    return clean_text(data)


def create_output(query,query_emb,doc_emb,doc_text, file_names):
    #builds pandas df that is later used to generate final output 
    doc_emb = doc_emb.reshape(-1, 768)
    scores = np.matmul(query_emb, doc_emb.transpose(1,0))[0].tolist() #MIPS
    doc_score_pairs = list(zip(doc_text, scores, file_names))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    k = 10
    
    probs = softmax(sorted(scores,reverse = True)[:k])
    table = {"Answer":[],"Passage":[],"Probabilities":[],"Source":[]}
    
    for i, (passage, _, names) in enumerate(doc_score_pairs[:k]):
        passage = clean_text(passage)
        
        if probs[i] > 0.1 or (i < 3 and probs[i] > 0.05): #generate answers for more likely passages but no less than 2
            QA = {'question':query,'context':passage}
            ans = pipe(QA)
            probabilities = """P(aIp): {}, P(aIp,q): {}, P(pIq): {}""".format(round(ans["score"],4), round(ans["score"]*probs[i],4), round(probs[i],4))
            passage = passage.replace(str(ans["answer"]),str(ans["answer"]).upper()) 
            table["Passage"].append(passage)
            table["Answer"].append(str(ans["answer"]).upper())
            table["Probabilities"].append(probabilities)
            table["Source"].append(names)
        else:
            table["Passage"].append(passage)
            table["Answer"].append("no_answer_calculated")
            table["Probabilities"].append("P(pIq): {}".format(round(probs[i],4)))
            table["Source"].append(names)
            
    df = pd.DataFrame(table)
    return df

def load_metadata():
    if "metadata.json" not in os.listdir():
        save_metadata({"last_update":datetime.now().strftime("%d-%m-%Y %H:%M:%S")}) #make sure there is always one
    with open('metadata.json', 'r') as f:
        metadata = json.load(f) 
    return metadata

def save_metadata(metadata):
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f)

def save_text(meta_key,doc_name,text):
    if meta_key not in os.listdir("docs_text"):
        os.mkdir("docs_text/{}".format(meta_key))
        with open("docs_text/{}/{}.txt".format(meta_key,doc_name),"w+",encoding ="utf-8") as f:
            f.write(text)
            f.close()

def check_log_history(query,doc_name,text,metadata=None):
    #check if query was asked for the same document. 
    #If true then recover it (also checks if answer was saved)
    if metadata is None:
        metadata = load_metadata()
    meta_key = str(hashlib.sha256(text.encode()).hexdigest()) #this will 'always' be unique
    q_a = str(hashlib.sha256(str([query,doc_name]).encode()).hexdigest())+".csv" #this one will help retrieving answer
    
    if meta_key in metadata.keys() and meta_key in os.listdir("HISTORY"):
        if q_a in metadata[meta_key].keys():
            return q_a, True
    
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    metadata.update({meta_key:{q_a:{"query":query,
                                    "doc_name":doc_name,
                                    "time":current_time}}})
    if "merge" not in query:
        save_text(meta_key,doc_name,text) #because why not
    save_metadata(metadata) 
    return q_a, False

def build_oneans(row):
    answer,passage,probs,source = row
    one_ans = """
|{}|{}|{}|{}|""".format(answer,passage,probs,source,)
    return one_ans

def build_table(df):
    #build final markdown table to be displayed by streamlit
    table = """
|Answer      |Passage |Probabilities     | Source|
| :---:       |    :---:   |    :---:   |  :---:   |"""
    
    for i in range(len(df)):
        table+= build_oneans(tuple(df.loc[i].values))
    return table
