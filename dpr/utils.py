"""
Utility functions required by pages to load 'interesting' informations and 
maintaining directory
"""

import os
from datetime import datetime
from dpr import core

def get_lastupdate():
    if "metadata.json" not in os.listdir():
        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        return current_time, False
    else:
         return core.load_metadata()["last_update"], True

def create_dirs():
    for needed_dir in ["HISTORY","single_doc","merged_docs","docs_dump","docs_text","youtube_videos"]:
        if needed_dir not in os.listdir():
            os.mkdir(needed_dir)
    if "clusters" not in os.listdir("merged_docs"):
        os.mkdir("merged_docs/clusters")
    
def merged_info():
    if "emb.npy" in os.listdir("merged_docs"):
        size = os.path.getsize("merged_docs/emb.npy")
    else:
        size = 0
    return int(size/3131), round(size/(1024**3),4)

def single_info():
    num_encoded = len(os.listdir("single_doc"))
    size = sum(os.path.getsize('single_doc/'+f) for f in os.listdir('single_doc'))
    return num_encoded,round(size/(1024**3),4)

def pdf_dump_info():
    pass

    
