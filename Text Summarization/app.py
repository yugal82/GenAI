import os
from dotenv import load_dotenv
load_dotenv()

import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# streamlit app
st.set_page_config(page_title="Summarize your Youtube videos or any webpage", page_icon="🤖")
st.title("🤖The AI tool to easily summarize text from YouTube video or webpage🤖")

groq_api_key = st.sidebar.text_input(label="Please enter your GROQ API key.", type="password")
options = ["Gemma2-9b-It", "Llama3-8b-8192", "Mixtral-8x7b-32768"]
selected_model = st.sidebar.selectbox(label="Please select an Open Source Model", options=options)
url = st.text_input(label="Enter URL of YouTube video or webpage.", label_visibility="collapsed")

# LLM model
llm_model = ChatGroq(model=selected_model, groq_api_key=groq_api_key)

# Prompt template
prompt_template = """
Provide a summary of the following content in 300-400 words
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])


if st.button("Summarize the content"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please enter the required fields.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Loading..."):
                # loading the website/yt data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url], 
                        ssl_verify=True,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                
                docs = loader.load()
                
                # chain for summarization
                chain = load_summarize_chain(llm=llm_model, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)
                
                st.success(summary)
        except Exception as e:
            st.error(f"Error occurred while running: {e}")
            
                    