import os
from constants import openai_key

os.environ['OPENAI_API_KEY'] = openai_key

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain

demo_template = ''' I want you to act as an Financial Advisor for people. In an easy way, explain the basis of {financial_concept}.'''

prompt = PromptTemplate(
    input_variables = ['financial_concept'],
    template = demo_template
    )

prompt.format(financial_concept='income tax')

llm = OpenAI(temperature=0.7)

chain = LLMChain(llm=llm, prompt=prompt)

out = chain.run('income tax')

print(out)


#Language Translation 

template = ''' In an easy way translate the following sentence {sentence} into {target_language}.'''

language_prompt = PromptTemplate(
    input_variables = ['sentence', 'target_language'],
    template = template
    )

language_prompt.format( sentence ='income tax', target_language='hindi')

language_chain = LLMChain(llm=llm, prompt=language_prompt)

tl = language_chain({'sentence': "donkey", 
                     'target_language' : "hindi"})

print(tl)

#FewShotPromptTemplate