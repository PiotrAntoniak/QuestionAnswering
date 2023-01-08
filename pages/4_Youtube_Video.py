import streamlit as st
from dpr import youtube
from dpr import core
import time


st.sidebar.markdown("# Youtube Video")
st.markdown("# Youtube Video")

st.markdown("""
            This tab makes use of youtube videos - input a link, ask a question and wait for your answer!
            """)

link = None

#get question and yt that is uspposed to have an answer
query = st.text_input("What is your question?")
link = st.text_input("Which youtube video (please insert the link) would you like to use?")


if st.button('Predict'):
    st.write("Using: "+link+" to answer your question")
    st.video(link)
    start = time.time()
    results = youtube.get_answer(query,link)
    st.write("It took: "+ str(time.time()-start) +" seconds to get the answer")
    st.markdown(core.build_table(results))


