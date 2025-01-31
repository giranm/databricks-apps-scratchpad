import os
import logging

from dotenv import load_dotenv

import gradio as gr

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

from libs.genie import GenieHandler

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
ENVIRONMENT = os.getenv("ENVIRONMENT", "PROD")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

# Initialize WorkspaceClient appropriately for the environment
config: Config = None
wc: WorkspaceClient = None
genie_handler: GenieHandler = None

if ENVIRONMENT == "LOCAL":
    config = Config(profile="DEFAULT")
    wc = WorkspaceClient(config=config)
else:
    wc = WorkspaceClient()


def init_state():
    """Initialize the state with empty token"""
    return {
        "user_token": "",
        "genie_rooms": [],
        "selected_genie_room_id": "",
        "current_conversation_id": "",
    }


def update_token(token, state):
    """Update the user token in state while preserving other state values"""
    return {**state, "user_token": token}


def update_genie_rooms(state):
    """Update the genie rooms in state while preserving other state values"""
    genie_handler = GenieHandler(
        databricks_host=DATABRICKS_HOST,
        databricks_user_token=state["user_token"],
    )

    genie_rooms = []
    try:
        genie_rooms = genie_handler.get_genie_rooms()["data_rooms"]
        logger.info(f"Genie rooms: {genie_rooms}")
    except Exception as e:
        logger.error(f"Error getting genie rooms: {str(e)}", exc_info=True)

    return {**state, "genie_rooms": genie_rooms}


def update_selected_genie_room_id(genie_room_name, state):
    """Update the selected genie room id in state while preserving other state values"""
    genie_room_id = next(
        (
            room["space_id"]
            for room in state["genie_rooms"]
            if room["display_name"] == genie_room_name
        ),
        None,
    )
    logger.info(f"Selected genie room name: {genie_room_name} and id: {genie_room_id}")
    return {**state, "selected_genie_room_id": genie_room_id}


def update_current_conversation_id(current_conversation_id, state):
    """Update the current conversation id in state while preserving other state values"""
    return {**state, "current_conversation_id": current_conversation_id}


def message_handler(message, history, state):
    """
    Query the LLM with the given message and chat history using ChatDatabricks.
    """

    genie_handler = GenieHandler(
        databricks_host=DATABRICKS_HOST,
        databricks_user_token=state["user_token"],
    )

    if not state["user_token"]:
        logger.warning("User token is not set")
        return "Please enter your user token first"

    if not state["selected_genie_room_id"]:
        logger.warning("Genie room is not selected")
        return "Please select a Genie room first"

    if not message.strip():
        logger.warning("User question is empty")
        return "ERROR: The question should not be empty"

    message_content = None
    handler_response = None

    try:
        logger.info(f"Message: {message}")

        if not state["current_conversation_id"]:
            logger.info("Starting a new conversation")
            handler_response = genie_handler.start_conversation(
                state["selected_genie_room_id"], message
            )
        else:
            logger.info(
                f"Continuing conversation id: {state['current_conversation_id']}"
            )
            handler_response = genie_handler.create_message(
                state["selected_genie_room_id"],
                state["current_conversation_id"],
                message,
            )

        text = (
            handler_response.get("attachments")[0].get("text").get("content")
            if handler_response.get("attachments")[0].get("text")
            else None
        )
        query_description = (
            handler_response.get("attachments")[0].get("query").get("description")
            if handler_response.get("attachments")[0].get("query")
            else None
        )
        if query_description:
            message_id = handler_response.get("id")
            query_result = genie_handler.get_query_result(
                space_id=state["selected_genie_room_id"],
                conversation_id=state["current_conversation_id"],
                message_id=message_id,
            )
            table_as_str = genie_handler.prettify_query_result(query_result)
            message_content = f"{query_description}\n" f"```{table_as_str}```"
        elif text:
            message_content = f"{text}"

        return message_content

    except Exception as e:
        logger.error(f"Error querying model: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks() as demo:

    # Initialize state
    state = gr.State(init_state())

    # Render components
    gr.Markdown("# AI/BI Genie on Gradio")
    gr.Markdown("Ask questions and get responses from AI/BI Genie")

    with gr.Row():
        token_input = gr.Textbox(
            type="password",
            label="Databricks Token",
            placeholder="Enter your Databricks token",
            show_label=True,
            scale=4,
        )
        set_token_btn = gr.Button("Set Token", scale=1)

    # Container for components that should only show after token is set
    with gr.Group(visible=False) as authenticated_container:
        genie_rooms_dropdown = gr.Dropdown(
            label="Select a Genie room",
            choices=[],  # Start with empty choices
            show_label=True,
        )

        chatbot = gr.ChatInterface(
            fn=message_handler,
            additional_inputs=[state],
        )

    # Handle input changes
    def on_token_set(token, state):
        """Handle token setting and show/hide components"""
        state_with_token = update_token(token, state)
        state_with_rooms = update_genie_rooms(state_with_token)

        # Get room choices from updated state
        room_choices = [
            room["display_name"]
            for room in state_with_rooms["genie_rooms"]
            if room["display_name"]
        ]

        return (
            state_with_rooms,  # Update state
            gr.update(visible=True),  # Show authenticated container
            gr.update(choices=room_choices),  # Update dropdown choices
        )

    set_token_btn.click(
        fn=on_token_set,
        inputs=[token_input, state],
        outputs=[
            state,
            authenticated_container,
            genie_rooms_dropdown,
        ],
    )

    genie_rooms_dropdown.change(
        fn=update_selected_genie_room_id,
        inputs=[genie_rooms_dropdown, state],
        outputs=[state],
    )

# Entrypoint
if __name__ == "__main__":
    demo.launch()
