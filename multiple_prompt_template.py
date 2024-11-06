#integrate our code with openai api 
import os
from constants import openai_key
from langchain_community.llms import OpenAI
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain


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



chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person')

#2nd Prompt Templates
# second_input_prompt = PromptTemplate(
#     input_variables=['person'], 
#     template = "when was {person} born"
# ) 
second_input_prompt = PromptTemplate(
    input_variables=['name'], 
    template = "when was {name} born"
    )

chain2 = LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob')

third_input_prompt = PromptTemplate(
    input_variables=['dob'], 
    template = "mention 5 major events that happend when {dob} in the world"
) 

chain3 = LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='description')

# parent_chain = SimpleSequentialChain(chains = [chain, chain2], verbose=True)
parent_chain = SequentialChain(
    chains = [chain, chain2,chain3], input_variables=['name'], output_variables = ['person','dob','description'],verbose=True)

if input_text:
    # SimpleSequentialChain
    # st.write(parent_chain.run(input_text))
    
    #sequentialchain, write the inputs in key value pairs
    st.write(parent_chain({'name' : input_text}))
    
    