import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API"] = os.getenv("LANGCHAIN_API")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

# basic pipeline to make any GenAI application: Data ingestion--->Text Splitting--->Text Embedding--->VectorstoreDb--->Retrieval Chain

import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions asked"),
        ("user", "Question: {question}")
    ]
)

st.title("Langchain demo with Google Gemma LLM model")
input_text = st.text_input("What question do you have in mind?")

# Ollama Gemma model
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
    
    
# we here dont follow the pipeline of 4 steps as we directly use one of the LLM models which are pre-trained on billions of parameters. So when we ask a question, we directly get the answer to the questions