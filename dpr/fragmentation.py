"""
Functions required by 003_Multiple Documents.py. 
They introduce kmeans to improve speed when the emb.py file gets too big.
"""

import numpy as np
import hashlib
from sklearn.cluster import KMeans
from tqdm import tqdm

def optimal_k():
    dicto = np.load('merged_docs/files.npy',allow_pickle='TRUE').item()
    num_keys = len(dicto.keys())
    return int(np.sqrt(num_keys))

def load_emb():
    doc_emb = np.load('merged_docs/emb.npy',allow_pickle='TRUE').item()
    doc_text = np.load('merged_docs/spans.npy',allow_pickle='TRUE').item()
    file_names_dicto = np.load('merged_docs/files.npy',allow_pickle='TRUE').item()
    
    emb = np.array(list(doc_emb.values()))
    spans =  np.array(list(doc_text.values()))
    names =  np.array(list(file_names_dicto.values()))
    
    return emb,spans,names

def index_cluster(emb,spans,names,idx):
    return emb[idx],spans[idx],names[idx]

def cluster():
    dicto = {}
    k = optimal_k()
    
    emb,spans,names = load_emb()
    kmeans = KMeans(n_clusters=k , random_state=2137).fit(emb)
    for i in tqdm(kmeans.labels_):
        idx = kmeans.labels_ == i
        temp_cluster = index_cluster(emb,spans,names,idx)
        temp_center = kmeans.cluster_centers_[i]
        
        name = str(hashlib.sha256(str(temp_center).encode()).hexdigest())
        #a trick to use np.save on tuple
        np.save('merged_docs/clusters/{}'.format(name),{"key":temp_cluster})
        
        dicto.update({name:temp_center})
        
    np.save('merged_docs/clusters/clusters.npy',dicto)
        
def load_cluster(query_emb):
    dicto = np.load('merged_docs/clusters/clusters.npy',allow_pickle='TRUE').item()
    clusters = np.array(list(dicto.values()))
    names = np.array(list(dicto.keys()))
    scores = np.matmul(query_emb, clusters.transpose(1,0))[0].tolist()
    
    doc_score_pairs = list(zip(names, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    
    cluster_to_load = doc_score_pairs[0][0]+".npy"
    cluster = np.load('merged_docs/clusters/'+cluster_to_load,allow_pickle='TRUE').item()
    return cluster["key"] #this trick
    
    
    
    
    
    
    
    
    
    
    
    