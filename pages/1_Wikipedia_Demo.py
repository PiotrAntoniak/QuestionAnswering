import streamlit as st
from dpr import wiki
from dpr import core
import time
st.sidebar.markdown("# Wikipedia Demo")
st.markdown("# Wikipedia Demo")

st.markdown("""
            This tab utilizes wikipedia to answer your question. Type in the name of the article
            i.e. 'Barrack Obama' and ask your question: i.e. 'Who is Barrack Obama?'.
            """)

wikipage = None

#get question and wikipage that is uspposed to have an answer
query = st.text_input("What is your question?")
wikipage = st.text_input("Which wikipedia page would you like to consult?")


if st.button('Predict'):
    st.write("Using: "+wikipage+" page to answer your question")
    #start = time.time()
    results = wiki.get_answer(query,wikipage)
    #st.write("It took: "+ str(time.time()-start) +" seconds to get the answer")
    core.build_table(results)
