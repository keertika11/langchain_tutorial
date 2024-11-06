import os
import logging
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from constants import openai_key

# Set up logging to display information in the terminal
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = openai_key

# Initialize the OpenAI model with a specified temperature
llm = OpenAI(temperature=0.8)

# Define the prompt template for evaluating relevance
relevance_prompt = PromptTemplate(
    input_variables=["query", "point"],
    template=(
        "User Query: {query}\n\n"
        "Point: {point}\n\n"
        "Is this point relevant to the user's query? Respond with 'Yes' or 'No'."
    )
)

# Define the prompt template for summarization
summarization_prompt = PromptTemplate(
    input_variables=["query", "points"],
    template=(
        "User Query: {query}\n\n"
        "Relevant Points:\n{points}\n\n"
        "Provide a concise summary that addresses the user's query based on the relevant points."
    )
)

# Create the LLMChains for relevance evaluation and summarization
relevance_chain = LLMChain(llm=llm, prompt=relevance_prompt)
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

def evaluate_relevance(query, point):
    """
    Evaluates the relevance of a point to the user's query.

    Args:
        query (str): The user's query.
        point (str): The point to evaluate.

    Returns:
        bool: True if the point is relevant, False otherwise.
    """
    logger.info(f"Evaluating relevance of point: '{point}' to query: '{query}'")
    response = relevance_chain.run({"query": query, "point": point}).strip().lower()
    logger.info(f"Model response: {response}")
    is_relevant = response == 'yes'
    logger.info(f"Point relevance: {'Relevant' if is_relevant else 'Not Relevant'}\n")
    return is_relevant

def process_query(query, points):
    """
    Processes the user's query by selecting relevant points and providing an appropriate response.

    Args:
        query (str): The user's query.
        points (list): A list of points to evaluate.

    Returns:
        str: A response addressing the user's query.
    """
    logger.info(f"Processing query: '{query}'\n")
    # Evaluate each point for relevance
    relevant_points = []
    for point in points:
        if evaluate_relevance(query, point):
            relevant_points.append(point)

    if not relevant_points:
        logger.info("No relevant points found.\n")
        return "No relevant information found for your query."

    if len(relevant_points) == 1:
        logger.info("One relevant point found. Returning the point.\n")
        return relevant_points[0]

    # Format the relevant points as a numbered list
    formatted_points = "\n".join([f"{i+1}. {point}" for i, point in enumerate(relevant_points)])
    logger.info(f"Multiple relevant points found:\n{formatted_points}\n")
    # Generate a summary based on the relevant points
    summary = summarization_chain.run({"query": query, "points": formatted_points}).strip()
    logger.info(f"Generated summary:\n{summary}\n")
    return summary

# Example usage
if __name__ == "__main__":
    user_query = "What are the benefits of adopting cloud computing for businesses?"
    key_points = [
        "Scalability: Cloud services allow businesses to scale resources up or down based on demand.",
        "Cost Efficiency: Reduces the need for significant capital expenditure on hardware.",
        "Accessibility: Enables access to services and data from any location with internet connectivity.",
        "Data Security: Cloud providers offer advanced security features to protect data.",
        "Environmental Impact: Cloud computing can lead to more efficient energy use."
        "Historical Context: The concept of cloud computing dates back to the 1960s.",
        "Personal Anecdote: My friend recently migrated his business to retail"
    ]

    response = process_query(user_query, key_points)
    logger.info("Final Response:")
    logger.info(response)
