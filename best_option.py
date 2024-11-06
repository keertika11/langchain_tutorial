import os
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from constants import openai_key

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = openai_key

# Initialize the OpenAI model
llm = OpenAI(temperature=0.8)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["options"],
    template="Given the following options:\n{options}\nPlease select the best option and explain your choice."
)

# Create the LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)

def select_best_option(options):
    # Format the options as a numbered list
    formatted_options = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
    print(formatted_options)
    # Run the chain with the formatted options
    result = chain.run({"options": formatted_options})
    return result

# Example usage
options = [
    "Option A: High quality but expensive.",
    "Option B: Moderate quality and affordable.",
    "Option C: Low quality but very cheap."
]

best_option = select_best_option(options)
print(best_option)
