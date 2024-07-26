import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot"

# prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assitant. Please give appropriate response to the user queries."),
        ("user", "Question: {question}")
    ]
)


# for every LLM model, we have a temperature value between 0 to 1. 0 means the model will not be creative in giving us the answer, and 1 means the model will be creative and give different results everytime.
def generate_response(question, temperature, max_tokens, model):
    llm_model = Ollama(model=model)
    output_parser = StrOutputParser()
    chain = prompt|llm_model|output_parser
    answer = chain.invoke({'question': question})
    return answer
    
# title of the app
st.title("Enhanced QA chatbot")

# st.sidebar.text_input("Enter your API key", type="password")
options = ["gemma:2b", "llama2", "mistral"]
model = st.sidebar.selectbox(label="Select open source models", options=options)
temperature = st.sidebar.slider(label="Select temperature value", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider(label="Select temperature value", min_value=50, max_value=300, value=150)

# main interface
st.write("Go ahead and ask any questions")
question = st.text_input("You:")

if question:
    response = generate_response(question, temperature, max_tokens, model)
    st.write(response)
else:
    st.write("Please provide a query")