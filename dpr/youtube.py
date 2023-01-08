from dpr import core
from dpr import singledoc
from pytube import YouTube
import os
import pandas as pd

def check_if_downloaded(file_name):
    if file_name in os.listdir("youtube_videos/"):
        return True

def download_video(link):
    yt_id = (link.split("="))[1].split("&")[0]
    if check_if_downloaded(yt_id):
        path = "youtube_videos/yt_{}.mp4".format(yt_id)
    else:
        path = YouTube(link).streams.filter(only_audio=True)[0].download(filename="youtube_videos/yt_{}.mp4".format(yt_id))
    
    return path

def encoded_video_text(link):
    yt_id = (link.split("="))[1].split("&")[0]
    file_name = "yt_{}.mp4".format(yt_id)
    
    if singledoc.check_if_encoded(file_name):
        doc_emb,spans,file_names,text = singledoc.load_encoded(file_name+".npy")
    else:
        path = download_video(link)
        
        text = core.transcribe_video(path)
        doc_emb, spans, file_names,text = core.encode_docs((file_name,text),maxlen = 64, stride = 32)
        singledoc.save_encoded(doc_emb, spans, file_names,file_name+".npy",text)
   
    return doc_emb, spans, file_names,text

def predict(query,doc_emb, spans, file_names):
    query_emb = core.encode_query(query)
    df = core.create_output(query,query_emb,doc_emb,spans,file_names)
    return df

def get_answer(query,link):
    yt_id = (link.split("="))[1].split("&")[0]
    file_name = "yt_{}.mp4".format(yt_id)
    
    doc_emb, spans, file_names,text = encoded_video_text(link)
    
    path,flag = core.check_log_history(query,file_name,text)
    
    if not flag:
         df = predict(query,doc_emb, spans, file_names)
         df.to_csv("HISTORY/{}".format(path),index=False)
    else:
        try:
            #in case the read fails
            df = pd.read_csv("HISTORY/{}".format(path))
        except:
            #just act as if it's new query
            #st.write(e)
            df = predict(query,doc_emb, spans, file_names)
            df.to_csv("HISTORY/{}".format(path),index=False)
    return df