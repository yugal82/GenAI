import streamlit as st
import pandas as pd
import numpy as np

## Title of the application
st.title("Streamlit Application")

name = st.text_input("Enter your good name:")
if name:
    st.write(f"Hello {name}!")
    

age = st.slider("Select your age: ", 0, 100, 18)
st.write(f"Your age: {age}")
    
options = ["Python", "C++", "Javascript", "Java"]
choice = st.selectbox("Choose your favorite programming language", options, index=None)
st.write(f"Your favorite programming language is: {choice}")