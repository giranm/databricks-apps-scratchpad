import os
import logging

from dotenv import load_dotenv

import gradio as gr

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
        "user_authenticated": False,
        "genie_rooms": [],
        "selected_genie_room_id": "",
        "selected_genie_room_name": "",
        "current_conversation_id": "",
        "current_curated_questions": [],
    }


def update_and_validate_token(state, token):
    """Update and validate the user token in state while preserving other state values"""
    redacted_token = "*" * (len(token) - 4) + token[-4:] if len(token) > 4 else "****"
    logger.info(f"Validating token: {redacted_token}")
    res = requests.get(
        url=f"https://{DATABRICKS_HOST}/api/2.0/token/list",
        headers={"Authorization": f"Bearer {token}"},
    )
    if res.status_code == 200:
        logger.info(f"Token is valid")
        gr.Success("User has been authenticated", duration=5)
        return {**state, "user_token": token, "user_authenticated": True}
    else:
        logger.error(f"Error validating token: {res.text}")
        raise gr.Error(
            f"Error validating token: {res.text}", print_exception=False, duration=5
        )


def update_genie_rooms(state):
    """Update the genie rooms in state while preserving other state values"""
    genie_handler = GenieHandler(
        databricks_host=DATABRICKS_HOST,
        databricks_user_token=state["user_token"],
    )

    genie_rooms = []
    try:
        genie_rooms = genie_handler.get_genie_rooms()
        logger.info(f"Genie rooms: {genie_rooms}")
    except Exception as e:
        logger.error(f"Error getting genie rooms: {str(e)}", exc_info=True)
        raise gr.Error(
            f"Error getting genie rooms: {str(e)}", print_exception=False, duration=5
        )

    return {**state, "genie_rooms": genie_rooms}


def update_selected_genie_room_id(state, genie_room_name):
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
        curated_questions = genie_handler.get_curated_questions(genie_room_id)
        if curated_questions:
            current_curated_questions.extend(curated_questions)
        logger.info(f"Current curated questions: {current_curated_questions}")
    except Exception as e:
        logger.error(f"Error getting curated questions: {str(e)}", exc_info=True)
        raise gr.Error(
            f"Error getting curated questions: {str(e)}",
            print_exception=False,
            duration=5,
        )

    # Handle if existing conversation is found from switching rooms - if so close it.
    if state["current_conversation_id"]:
        logger.info(
            f"Closing existing conversation id: {state['current_conversation_id']}"
        )
        state["current_conversation_id"] = None

    return {
        **state,
        "selected_genie_room_id": genie_room_id,
        "selected_genie_room_name": genie_room_name,
        "current_curated_questions": current_curated_questions,
    }


def message_handler(message, history, state):
    """
    Query Databricks AI/BI Genie using the user message and chat history.
    """

    genie_handler = GenieHandler(
        databricks_host=DATABRICKS_HOST,
        databricks_user_token=state["user_token"],
    )

    if not state["user_token"]:
        logger.warning("User token is not set")
        gr.Warning(
            "Please enter a valid user token", duration=5
        )  # Incase of DOM overrides
        return "Warning: Please enter a valid user token", state

    if not state["selected_genie_room_id"]:
        logger.warning("Genie room is not selected")
        gr.Warning(
            "Please select a Genie room first", duration=5
        )  # Incase of DOM overrides
        return "Warning: Please select a Genie room first", state

    if not message.strip():
        logger.warning("User question is empty")
        gr.Warning("The question should not be empty", duration=5)
        return "Warning: The question should not be empty", state

    message_content = None
    handler_response = None

    try:
        logger.info(
            f"User input: `{message}` under room ID: {state['selected_genie_room_id']}"
        )

        if not state["current_conversation_id"]:
            logger.info(
                f"Starting a new conversation with Genie under room ID: {state['selected_genie_room_id']}"
            )
            handler_response = genie_handler.start_conversation(
                state["selected_genie_room_id"], message
            )
            state["current_conversation_id"] = handler_response["conversation_id"]
            logger.info(
                f"Created new conversation with ID: {state['current_conversation_id']} under room ID: {state['selected_genie_room_id']}"
            )
        else:
            logger.info(
                f"Continuing conversation id: {state['current_conversation_id']} under room ID: {state['selected_genie_room_id']}"
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
        logger.error(f"Error handling message: {e}", exc_info=True)
        return f"Error: {e}", state


"""
Gradio Interface
"""

with gr.Blocks(
    title="AI/BI Genie on Gradio",
    css="""    
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
        '<div class="warning-box">⚠️ This app uses experimental features of Databricks AI/BI Genie and is meant for demonstrational purposes only.</div>',
        elem_classes="warning-container",
    )

    # Token Input section
    with gr.Row() as token_row:
        token_input = gr.Textbox(
            type="password",
            label="Databricks Personal Access Token",
            placeholder="Enter your Databricks token",
            show_label=True,
            scale=8,
        )
        set_token_btn = gr.Button(
            "Authenticate using Token", scale=1, size="lg", variant="primary"
        )

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
    def on_room_select(state, genie_room_name):
        updated_state = update_selected_genie_room_id(state, genie_room_name)

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
        inputs=[state, genie_rooms_dropdown],
        outputs=[state, chat_row, suggestion_row] + suggestion_buttons,
    )

    # Suggestion handler
    def use_suggestion(suggestion_text):
        return suggestion_text

    # Set up suggestion buttons
    for btn in suggestion_buttons:
        btn.click(
            fn=use_suggestion,
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
            raise gr.Error("Token is required", print_exception=False, duration=5)

        state_with_validated_token = update_and_validate_token(state, token)
        state_with_rooms = update_genie_rooms(state=state_with_validated_token)

        room_choices = [
            room["display_name"]
            for room in state_with_rooms["genie_rooms"]
            if room["display_name"]
        ]
        return [
            state_with_rooms,
            gr.update(visible=False),  # Hide token row
            gr.update(visible=True),  # Show room selection
            gr.update(choices=room_choices),
        ]

    # Handle token set
    set_token_btn.click(
        fn=on_token_set,
        inputs=[token_input, state],
        outputs=[
            state,
            token_row,  # Add token_row to outputs
            room_selection_row,  # Add room_selection_row to outputs
            genie_rooms_dropdown,
        ],
    )

# Entrypoint
if __name__ == "__main__":
    demo.launch()
