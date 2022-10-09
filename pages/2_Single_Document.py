import streamlit as st
from dpr import singledoc
from dpr import core
import time
st.sidebar.markdown("# Single Document")
st.markdown("# Single Document question answering")

st.markdown(""""
            This tab is able to answer your question by using a single pdf or docx document. 
            Just drag it below and type in your question. The procedure is optimized so asking a second 
            question with the same document will be way faster than the first one!
            """)


data = None
query = st.text_input("What is your question?")
data = st.file_uploader("Upload your pdf!",type="pdf")

if st.button('Predict'):
    if data is not None:
        doc_name = data.name
        st.write("Using file: "+doc_name+" to answer your question")
        start = time.time()
        results = singledoc.get_answer(query,data,doc_name)
        st.write("It took: "+ str(time.time()-start) +" seconds to get the answer")
        st.markdown(core.build_table(results))

