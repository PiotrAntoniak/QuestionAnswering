#landing page
import streamlit as st
st.sidebar.markdown("# Question Answering")
st.markdown("# Question Answering from Documents using Deep Passage Retrieval")
st.markdown("""
            On the left you can see three tabs to choose from:
            - Wikipedia Demo,
            - Single Document,
            - Multiple Documents.
            All of those tabs utilize transformers to retrieve data from documents and 
            then answer your question.
            
            Wikipedia Demo uses wikipedia articles - give it an article name and your question 
            and watch it return answers.
            
            Single Document tab requires you too feed a document in .pdf or .docx format and
            your question. 
            
            Finaly, Multiple Documents do not require documents - just your question!
            This tab will try to answer the query using previously encoded documents from a 
            Single Document query or let you use all of the documents in a special path.
            
            For more detailed information head into each of the tabs on the left.
            
            Enjoy!            
            """)
