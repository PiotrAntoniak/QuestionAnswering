"""
Functions required by 001_Wikipedia Demo.py
"""
from dpr import core
import wikipedia

def wikipedia_page(wiki_page):
    return wikipedia.page(wiki_page).content

def predict(query,text,wiki_page):
    doc_emb, spans, file_names = core.encode_docs((wiki_page,text),maxlen = 32, stride = 16)
    query_emb = core.encode_query(query)
    df = core.create_output(query,query_emb,doc_emb,spans, file_names)
    return df

def get_answer(query,wiki_page):
    text = core.clean_text(wikipedia_page(wiki_page))
    df = predict(query,text,wiki_page)
    return df
