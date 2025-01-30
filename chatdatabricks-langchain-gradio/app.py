import os
import logging

from dotenv import load_dotenv

import gradio as gr

from databricks_langchain import ChatDatabricks

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file, else taken from app.yaml via Apps runttime
load_dotenv()
MODEL_ENDPOINT_NAME = os.getenv("MODEL_ENDPOINT_NAME", "databricks-dbrx-instruct")

# Initialize ChatDatabricks model - assumes that the workspace host and token are set in the environment
chat_model = ChatDatabricks(
    endpoint=MODEL_ENDPOINT_NAME, temperature=0.1, max_tokens=400
)


def query_llm(message, history):
    """
    Query the LLM with the given message and chat history using ChatDatabricks.
    """
    if not message.strip():
        return "ERROR: The question should not be empty"

    try:
        logger.info(f"Sending request to model endpoint: {MODEL_ENDPOINT_NAME}")
        logger.info(f"Message: {message}")
        response = chat_model.invoke(message)
        logger.info("Received response from model endpoint")
        logger.info(f"Response: {response}")
        return response.content

    except Exception as e:
        logger.error(f"Error querying model: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


# Create Gradio interface
demo = gr.ChatInterface(
    fn=query_llm,
    title="ChatDatabricks on Gradio",
    description=f"Ask questions and get responses from the LLM model {MODEL_ENDPOINT_NAME}.",
    examples=[
        "What is machine learning?",
        "What are Large Language Models?",
        "What is Databricks?",
    ],
)

# Entrypoint
if __name__ == "__main__":
    demo.launch()
