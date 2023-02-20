import streamlit as st
from dpr import mergeddoc
from dpr import core
import time
from dpr import utils
st.sidebar.markdown("# Multiple Documents")
st.markdown("# Multiple Documents question answering")

st.markdown("""
            This page is slighly different from previous one. Now, you dont specify a document.
            Instead, question can be answered using all previously encoded documents from
            Single DOcument tab or a documents dump in docs_dump path.
            Finally, there are two options:
                
                - Re-run merge,
                
                - Use kmeans - higher speed at the cost of quality;
                
            First one merges all embeddings  into a single matrix for all documents in dump and previously used documents.
            WIthout using it, all your questions will be answered using previously created merged file and will not include
            most recent documents.
            Second options offers a tradeoff between speed and qccuracy of results. When the number of encoded 
            documents is low, it may not make sense to use it, but when it increases, this method is reccomended.
            """)

data = None

query = st.text_input("What is your question?")

m_info = utils.merged_info()
s_info = utils.single_info()

st.write("""Without feeding new document, you can ask questions and they will be answered using 
         ~{} spans (each has ~240 characters) taking roughly {}GB of space created
         from {} unique documents.""".format(m_info[0],m_info[1],s_info[0]))

#rerun_merge asks whether recalcualte embeddings for multiple docs QA
#this will result in extracting already encoded docs and docs_dump (pdfs,docxs)
#and using them to create new emb.npy and clsutering it using Kmeans
         
#use_kmeans - utilize kmeans to extract relevant cluster of passages (lower quality higher speed)
rerun_merge = st.checkbox('Re-run merge')
use_kmeans = st.checkbox('Use kmeans - higher speed at the cost of quality')
if st.button('Predict'):
    doc_name,flag = utils.get_lastupdate()
    if rerun_merge or not flag:
        st.write("This may take a while.")
    else:
        st.write("Using endoded docs prior to: "+doc_name+" to answer your question")
    #start = time.time()
    results = mergeddoc.get_answer(query,doc_name,rerun_merge,use_kmeans)
    #st.write("It took: "+ str(time.time()-start) +" seconds to get the answer")
    core.build_table(results)
