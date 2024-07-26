import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableWithMessageHistory

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Conversational Chatbot"
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")

st.title("Conversational RAG with PDF upload and chat history")

embeddings = OllamaEmbeddings(model="gemma:2b")

# input for groq api
groq_api = st.text_input("Please enter your Groq API", type="password")
if groq_api:
    llm = ChatGroq(groq_api_key=groq_api, model="Gemma2-9b-It")
    
    session_id = st.text_input("Session ID", value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}
        
    uploaded_file = st.file_uploader("Upload pdf", type="pdf", accept_multiple_files=False)
    
    if uploaded_file:
        documents = []
        # read uploaded file as documets
        temppdf = f'./temp.pdf'
        with open(temppdf, 'wb') as f:
            f.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
            
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
    
        # split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 500)
        splits = text_splitter.split_documents(documents=documents)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vector_store.as_retriever()
        
        contextualized_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just formulate a it if needed and otherwise return it as is"
        )
        
        contextualized_q_prompt = ChatPromptTemplate(
            [
                ("system", contextualized_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_q_prompt)
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the questions. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise"
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        def get_session_history(session_id):
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        user_input = st.text_input("Enter question related to uploaded PDF.")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            st.success(response["answer"])
else:
    st.warning("Enter your GROQ API key")