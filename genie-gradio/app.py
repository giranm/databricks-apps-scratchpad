import os
import logging

from dotenv import load_dotenv

import gradio as gr

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

import requests

from libs.genie import GenieHandler

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")

# Initial variables
genie_handler: GenieHandler = None

"""
Handlers Methods
"""


def init_state():
    """Initialize the state with empty token"""
    return {
        "user_token": "",
        "genie_rooms": [],
        "selected_genie_room_id": "",
        "selected_genie_room_name": "",
        "current_conversation_id": "",
        "current_curated_questions": [],
    }


def update_and_validate_token(token, state):
    """Update and validate the user token in state while preserving other state values"""
    redacted_token = "*" * (len(token) - 4) + token[-4:] if len(token) > 4 else "****"
    logger.info(f"Validating token: {redacted_token}")
    res = requests.get(
        url=f"https://{DATABRICKS_HOST}/api/2.0/token/list",
        headers={"Authorization": f"Bearer {token}"},
    )
    if res.status_code == 200:
        logger.info(f"Token is valid")
        return {**state, "user_token": token}
    else:
        logger.error(f"Error validating token: {res.text}")
        raise gr.Error(f"Error validating token: {res.text}", print_exception=False)


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
    """Update the selected genie room id and current curated questions in state while preserving other state values"""
    genie_handler = GenieHandler(
        databricks_host=DATABRICKS_HOST,
        databricks_user_token=state["user_token"],
    )

    genie_room_id = next(
        (
            room["space_id"]
            for room in state["genie_rooms"]
            if room["display_name"] == genie_room_name
        ),
        None,
    )
    logger.info(f"Selected genie room name: {genie_room_name} and id: {genie_room_id}")

    current_curated_questions = [{"question_text": "Explain the data set"}]
    try:
        fetched_questions = genie_handler.get_curated_questions(genie_room_id)
        if "curated_questions" in fetched_questions:
            current_curated_questions.extend(fetched_questions["curated_questions"])
        logger.info(f"Current curated questions: {current_curated_questions}")
    except Exception as e:
        logger.error(f"Error getting curated questions: {str(e)}", exc_info=True)

    return {
        **state,
        "selected_genie_room_id": genie_room_id,
        "selected_genie_room_name": genie_room_name,
        "current_curated_questions": current_curated_questions,
    }


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
        return "Please enter your user token first", state

    if not state["selected_genie_room_id"]:
        logger.warning("Genie room is not selected")
        return "Please select a Genie room first", state

    if not message.strip():
        logger.warning("User question is empty")
        return "ERROR: The question should not be empty", state

    message_content = None
    handler_response = None

    try:
        logger.info(f"Message: {message}")

        if not state["current_conversation_id"]:
            logger.info("Starting a new conversation")
            handler_response = genie_handler.start_conversation(
                state["selected_genie_room_id"], message
            )
            state["current_conversation_id"] = handler_response["conversation_id"]
            logger.info(
                f"Created new conversation with ID: {state['current_conversation_id']}"
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

        logger.info(f"Genie Handler response: {handler_response}")
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
            logger.info(f"Query result: {query_result}")
            columns, rows = genie_handler.transform_query_result(query_result)
            message_content = [
                f"{query_description}",
                gr.Dataframe(
                    headers=columns,
                    value=rows,
                    row_count=len(rows),
                    col_count=(len(columns), "fixed"),
                ),
            ]
        elif text:
            message_content = [f"{text}"]

        return message_content, state

    except Exception as e:
        logger.error(f"Error querying model: {str(e)}", exc_info=True)
        return f"Error: {str(e)}", state


"""
Gradio Interface
"""

with gr.Blocks(
    title="AI/BI Genie on Gradio",
    css="""
    /* Make containers transparent */
    .gradio-container {background-color: transparent !important}
    .contain {background-color: transparent !important}
    .gap {background-color: transparent !important}
    
    /* Style suggestion buttons */
    .suggestion-row button {
        margin: 0.25rem !important;
        min-width: 200px !important;
    }

    /* Custom warning box */
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.375rem;
        border: 1px solid #ffeeba;
        margin: 1rem 0;
    }
    """,
) as demo:
    # Initialize state
    state = gr.State(init_state())

    # Header
    gr.Markdown("# AI/BI Genie on Gradio")
    gr.Markdown("Ask questions and get responses from AI/BI Genie")
    gr.Markdown(
        '<div class="warning-box">⚠️ This app uses experimental features of Databricks AI/BI Genie. Please do not use this in production until the Genie API is released.</div>',
        elem_classes="warning-container",
    )

    # Token Input (Always visible)
    with gr.Row():
        token_input = gr.Textbox(
            type="password",
            label="Databricks Personal Access Token",
            placeholder="Enter your Databricks token",
            show_label=True,
            scale=8,
        )
        set_token_btn = gr.Button("Authenticate using Token", scale=1, size="lg")

    # Genie Room Selection (Initially hidden)
    with gr.Row(visible=False) as room_selection_row:
        genie_rooms_dropdown = gr.Dropdown(
            label="Select a Genie room", choices=[], show_label=True, scale=1
        )

    # Chat Interface (Initially hidden)
    with gr.Row(visible=False) as chat_row:
        # Chat handler
        def respond(message, history, state):
            bot_message, updated_state = message_handler(message, history, state)
            return bot_message, updated_state

        chatbot = gr.ChatInterface(
            fn=respond,
            type="messages",
            additional_inputs=[state],
            additional_outputs=[state],  # Add state as an output
        )

    # Curated Questions (Initially hidden)
    with gr.Row(visible=False, elem_classes="suggestion-row") as suggestion_row:
        with gr.Column(scale=1):
            gr.Markdown("### Suggested Questions")
            with gr.Row():
                MAX_SUGGESTIONS = 10
                suggestion_buttons = [
                    gr.Button(visible=False, size="md", scale=0)
                    for _ in range(MAX_SUGGESTIONS)
                ]

    # Room selection handler
    def on_room_select(genie_room_name, state):
        updated_state = update_selected_genie_room_id(genie_room_name, state)

        # Update suggestion buttons
        button_updates = []
        for i, btn in enumerate(suggestion_buttons):
            if i < len(updated_state["current_curated_questions"]):
                q = updated_state["current_curated_questions"][i]
                button_updates.append(gr.update(value=q["question_text"], visible=True))
            else:
                button_updates.append(gr.update(visible=False))

        # Show chat interface and suggestions
        return [
            updated_state,
            gr.update(visible=True),  # Update Row visibility only
            gr.update(visible=True),  # suggestion_row
        ] + button_updates

    # Room selection event
    genie_rooms_dropdown.change(
        fn=on_room_select,
        inputs=[genie_rooms_dropdown, state],
        outputs=[state, chat_row, suggestion_row] + suggestion_buttons,
    )

    # Suggestion handler
    def use_suggestion(suggestion_text):
        return suggestion_text

    # Set up suggestion buttons
    for btn in suggestion_buttons:
        btn.click(
            use_suggestion,
            inputs=[btn],
            outputs=[chatbot.textbox],  # Target the ChatInterface's textbox
        ).then(
            fn=None,
            js="""
            () => {
                // Find the submit button and click it
                document.querySelector('.submit-button').click();
                return [];
            }
            """,
        )

    # Token handler
    def on_token_set(token, state):

        if not token:
            logger.error("Token is required")
            raise gr.Error("Token is required", print_exception=False)

        state_with_token = update_and_validate_token(token, state)
        state_with_rooms = update_genie_rooms(state_with_token)

        room_choices = [
            room["display_name"]
            for room in state_with_rooms["genie_rooms"]
            if room["display_name"]
        ]

        return (
            state_with_rooms,
            gr.update(visible=True),  # Show room selection
            gr.update(choices=room_choices),
        )

    # Token event
    set_token_btn.click(
        fn=on_token_set,
        inputs=[token_input, state],
        outputs=[state, room_selection_row, genie_rooms_dropdown],
    )

# Entrypoint
if __name__ == "__main__":
    demo.launch()
