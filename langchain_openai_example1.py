#integrate our code with openai api 
import os
from constants import openai_key
from langchain_community.llms import OpenAI
import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

st.title("Langchain Demo with openai")
input_text = st.text_input("search the topic you want")

#openai LLMs
llm = OpenAI(temperature=0.8)

#Higher the Temperature, lesser the creative


if input_text:
    st.write(llm(input_text))