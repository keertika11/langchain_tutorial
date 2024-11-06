# LangChain Tutorial Scripts

This repository contains scripts following tutorials on LangChain. The tutorials can be found on YouTube:

- [LangChain Basics Tutorial](https://youtu.be/_FpT1cwcSLg?feature=shared)
- [Prompt Engineering with LangChain](https://youtu.be/t2bSApmPzU4?feature=shared)

## Environment Setup

Follow these steps to set up the environment:

1. **Create the Conda environment:**

   ```bash
   conda create -n lang_chain python=3.9
2. **Activate the environment:**
  
   ```bash
   conda activate lang_chain
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

## Script Descriptions

1. **langchain_openai_example1.py**
**Description**: Uses LangChain and openai_key to demonstrate a LangChain model on Streamlit.

2. **single_prompt_template.py**
   **Description**: Demonstrates how to use LLMChain with a single prompt template.

3. **multiple_prompt_template.py**
   **Description**:  Combines multiple prompts by taking the output of one prompt template as input for a second prompt template, using PromptTemplate and LLMChain.

4. **langchain_memory_conversation_openai.py**
   **Description**: Saves conversation history to memory for continuous conversation

5. **prompt_eng_langchain.py**
  **Description**: Covers prompt engineering techniques using LangChain.
