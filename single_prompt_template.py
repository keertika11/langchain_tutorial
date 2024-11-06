#integrate our code with openai api 
import os
from constants import openai_key
from langchain_community.llms import OpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


os.environ['OPENAI_API_KEY'] = openai_key

st.title("celebrity search results")
input_text = st.text_input("Enter the Celebrity Name")

#Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['name'], 
    template = "tell me about celebrity {name}"
)

#openai LLMs,Higher the Temperature, lesser the creative
llm = OpenAI(temperature=0.8)



chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True)


if input_text:
    st.write(chain.run(input_text))