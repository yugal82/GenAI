import streamlit as st
import pandas as pd
import numpy as np

## Title of the application
st.title("Streamlit Application")

## Display a simple text

st.write("This is some sample text")

df = pd.DataFrame({
    'first_column': [1,2,3,4],
    'second_column': [10,20,30,40]
})

st.write("DataFrame")
st.write(df)

## create a line chart
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
st.line_chart(chart_data)