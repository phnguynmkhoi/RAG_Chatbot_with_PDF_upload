import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter


from utils import write_uploaded_file
from prompts import contextualize_q_prompt, qa_prompt

import os
from dotenv import load_dotenv

def get_session_state(st, session_id) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

st.title('Q&A Chatbot with RAG and PDF file upload')
st.write("Upload PDFs and chat with their content")

api_key = st.text_input('Enter your Groq API key', type='password')

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model='Llama3-8b-8192')

    # input session
    session_id = st.text_input('Session ID', value='default')

    # stateful manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_file = st.file_uploader('Choose your PDF file', type='pdf', accept_multiple_files=True)

    if uploaded_file:
        documents = write_uploaded_file(uploaded_file, session_id)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
        new_docs = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(new_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # create history retriever chain
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        conversational_rag_chain = RunnableWithMessageHistory(rag_chain,
                                              get_session_state,
                                              history_messages_key='history',
                                              input_messages_key='input',
                                              output_messages_key='answer')

        user_input = st.text_input('Enter your query')
        if user_input:
            session_history=get_session_state(st, session_id)
            response = conversational_rag_chain.invoke({'input':user_input},
                                                       config={
                                                           'configurable': {
                                                               'session_id': session_id
                                                           }
                                                       })
            st.write(st.session_state.store)
            st.success('Assistant:' + response['answer'])
            st.write('Chat History: ', session_history)
            st.write(response)
else:
    st.write('Please provide the GROQ key api!')