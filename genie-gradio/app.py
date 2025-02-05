import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv

import gradio as gr

import requests

from libs.genie import GenieHandler

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


@dataclass
class AppState:
    """Class to handle application state"""

    user_token: str = ""
    user_authenticated: bool = False
    genie_rooms: List[Dict[str, Any]] = None
    selected_genie_room_id: str = ""
    selected_genie_room_name: str = ""
    current_conversation_id: str = ""
    current_curated_questions: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.genie_rooms is None:
            self.genie_rooms = []
        if self.current_curated_questions is None:
            self.current_curated_questions = []


class GenieGradioApp:
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        self.MAX_SUGGESTIONS = 10
        self.DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
        self.GRADIO_CONCURRENCY_COUNT = int(os.getenv("GRADIO_CONCURRENCY_COUNT", 20))
        self.GRADIO_MAX_THREADS = int(os.getenv("GRADIO_MAX_THREADS", 100))
        self.demo = self.create_demo()

    def create_demo(self) -> gr.Blocks:
        """Create and return the Gradio Blocks interface"""
        demo = gr.Blocks(
            title="AI/BI Genie on Gradio",
            css=self._get_css(),
        )

        with demo:
            # Initialize state for each session
            state = gr.State(AppState())  # Each user gets their own AppState

            # Create UI components
            self._create_header()
            token_row, token_input, set_token_btn = self._create_token_section()
            room_selection_row, genie_rooms_dropdown = self._create_room_selection()
            chat_row, chatbot = self._create_chat_interface(state)
            suggestion_row, suggestion_buttons = self._create_suggestions()

            # Setup event handlers
            self._setup_event_handlers(
                state,
                token_input,
                set_token_btn,
                token_row,
                room_selection_row,
                genie_rooms_dropdown,
                chat_row,
                suggestion_row,
                suggestion_buttons,
                chatbot,
            )

        return demo

    def validate_token(self, token: str) -> bool:
        """Validate Databricks token"""
        redacted_token = (
            "*" * (len(token) - 4) + token[-4:] if len(token) > 4 else "****"
        )
        self.logger.info(f"Validating token: {redacted_token}")

        response = requests.get(
            url=f"https://{self.DATABRICKS_HOST}/api/2.0/token/list",
            headers={"Authorization": f"Bearer {token}"},
        )
        return response.status_code == 200

    def handle_token_submission(
        self, token: str, state: AppState
    ) -> Tuple[AppState, gr.update, gr.update, gr.update]:
        """Handle token submission and validation"""
        if not token:
            self.logger.error("Token is required")
            raise gr.Error("Token is required", duration=5, print_exception=False)

        if not self.validate_token(token):
            self.logger.error(f"Invalid token")
            raise gr.Error("Invalid token", duration=5, print_exception=False)

        state.user_token = token
        state.user_authenticated = True
        self.logger.info(f"User authenticated successfully")
        gr.Success("User authenticated", duration=5)

        # Update Genie rooms
        genie_handler = GenieHandler(self.DATABRICKS_HOST, token)
        state.genie_rooms = genie_handler.get_genie_rooms()

        room_choices = [
            room["display_name"] for room in state.genie_rooms if room["display_name"]
        ]

        return (
            state,
            gr.update(visible=False),  # Hide token row
            gr.update(visible=True),  # Show room selection
            gr.update(choices=room_choices),
        )

    def handle_room_selection(self, state: AppState, room_name: str) -> List[Any]:
        """Handle room selection and update UI"""
        genie_handler = GenieHandler(self.DATABRICKS_HOST, state.user_token)

        room_id = next(
            (
                room["space_id"]
                for room in state.genie_rooms
                if room["display_name"] == room_name
            ),
            None,
        )

        state.selected_genie_room_id = room_id
        state.selected_genie_room_name = room_name
        state.current_conversation_id = ""
        self.logger.info(f"Selected Genie room: {room_name} (ID: {room_id})")

        # Get curated questions
        base_questions = [{"question_text": "Explain the data set"}]
        curated_questions = genie_handler.get_curated_questions(room_id) or []
        state.current_curated_questions = base_questions + curated_questions
        self.logger.info(
            f"Current curated questions: {state.current_curated_questions}"
        )

        # Update suggestion buttons
        button_updates = [
            (
                gr.update(
                    value=state.current_curated_questions[i]["question_text"],
                    visible=True,
                )
                if i < len(state.current_curated_questions)
                else gr.update(visible=False)
            )
            for i in range(self.MAX_SUGGESTIONS)
        ]

        return [
            state,
            gr.update(visible=True),  # chat_row
            gr.update(visible=True),  # suggestion_row
        ] + button_updates

    def handle_message(
        self, message: str, history: List[Tuple[str, str]], state: AppState
    ) -> Tuple[Any, AppState]:
        """Handle chat messages"""
        if not self._validate_chat_state(state, message):
            return "Please check your input and try again", state

        genie_handler = GenieHandler(self.DATABRICKS_HOST, state.user_token)

        try:
            response = self._process_message(genie_handler, message, state)
            return self._format_response(response, genie_handler, state), state
        except Exception as e:
            self.logger.error(f"Error handling message: {e}", exc_info=True)
            return f"Error: {e}", state

    def _validate_chat_state(self, state: AppState, message: str) -> bool:
        """Validate chat state before processing message"""
        if not state.user_token:
            self.logger.warning("Please enter a valid user token")
            gr.Warning("Please enter a valid user token", duration=5)
            return False
        if not state.selected_genie_room_id:
            self.logger.warning("Please select a Genie room first")
            gr.Warning("Please select a Genie room first", duration=5)
            return False
        if not message.strip():
            self.logger.warning("The input should not be empty")
            gr.Warning("The input should not be empty", duration=5)
            return False
        return True

    def _process_message(
        self, genie_handler: GenieHandler, message: str, state: AppState
    ) -> Dict[str, Any]:
        """Process the message and get response from Genie"""
        self.logger.info(
            f"Processing message: '{message}' under room ID: {state.selected_genie_room_id}"
        )

        response = None
        if not state.current_conversation_id:
            response = genie_handler.start_conversation(
                state.selected_genie_room_id, message
            )
            state.current_conversation_id = response["conversation_id"]
            self.logger.info(
                f"Created new conversation ID: {state.current_conversation_id} under room ID: {state.selected_genie_room_id}"
            )
        else:
            response = genie_handler.create_message(
                state.selected_genie_room_id,
                state.current_conversation_id,
                message,
            )

        self.logger.info(
            f"Received response from Genie: {response} under conversation ID: {state.current_conversation_id} and room ID: {state.selected_genie_room_id}"
        )
        return response

    def _format_response(
        self, response: Dict[str, Any], genie_handler: GenieHandler, state: AppState
    ) -> List[Any]:
        """Format the Genie response for display"""
        attachments = response.get("attachments", [{}])[0]

        # Handle text response
        if text := attachments.get("text", {}).get("content"):
            return [f"{text}"]

        # Handle query response
        if query := attachments.get("query"):
            query_description = query.get("description")
            if not query_description:
                return ["No query description available"]

            query_result = genie_handler.get_query_result(
                space_id=state.selected_genie_room_id,
                conversation_id=state.current_conversation_id,
                message_id=response.get("id"),
            )

            columns, rows = genie_handler.transform_query_result(query_result)
            return [
                f"{query_description}",
                gr.Dataframe(
                    headers=columns,
                    value=rows,
                    row_count=len(rows),
                    col_count=(len(columns), "fixed"),
                ),
            ]

        return ["No response content available"]

    def _get_css(self) -> str:
        """Get CSS styles for the UI"""
        return """    
        .suggestion-row button {
            margin: 0.25rem !important;
            min-width: 200px !important;
        }

        .warning-box {
            background-color: #fff3cd;
            color: #856404;
            padding: 1rem;
            border-radius: 0.375rem;
            border: 1px solid #ffeeba;
            margin: 1rem 0;
        }
        """

    def _create_header(self) -> None:
        """Create the header section"""
        gr.Markdown("# AI/BI Genie on Gradio")
        gr.Markdown("Ask questions and get responses from AI/BI Genie")
        gr.Markdown(
            '<div class="warning-box">⚠️ This app uses experimental features of '
            "Databricks AI/BI Genie and is meant for demonstrational purposes only.</div>",
            elem_classes="warning-container",
        )

    def _create_token_section(self) -> Tuple[gr.Row, gr.Textbox, gr.Button]:
        """Create the token input section"""
        with gr.Row() as token_row:
            token_input = gr.Textbox(
                type="password",
                label="Databricks Personal Access Token",
                placeholder="Enter your Databricks token",
                show_label=True,
                scale=8,
            )
            set_token_btn = gr.Button(
                "Authenticate using Token",
                scale=1,
                size="lg",
                variant="primary",
            )
        return token_row, token_input, set_token_btn

    def _create_room_selection(self) -> Tuple[gr.Row, gr.Dropdown]:
        """Create the room selection section"""
        with gr.Row(visible=False) as room_selection_row:
            genie_rooms_dropdown = gr.Dropdown(
                label="Select a Genie room",
                choices=[],
                show_label=True,
                scale=1,
            )
        return room_selection_row, genie_rooms_dropdown

    def _create_chat_interface(
        self, state: gr.State
    ) -> Tuple[gr.Row, gr.ChatInterface]:
        """Create the chat interface"""
        with gr.Row(visible=False) as chat_row:
            chatbot = gr.ChatInterface(
                fn=self.handle_message,
                additional_inputs=[state],
                additional_outputs=[state],
                type="messages",  # Add this to fix the deprecation warning
            )
        return chat_row, chatbot

    def _create_suggestions(self) -> Tuple[gr.Row, List[gr.Button]]:
        """Create the suggestions section"""
        with gr.Row(visible=False, elem_classes="suggestion-row") as suggestion_row:
            with gr.Column(scale=1):
                gr.Markdown("### Suggested Questions")
                with gr.Row():
                    suggestion_buttons = [
                        gr.Button(visible=False, size="md", scale=0)
                        for _ in range(self.MAX_SUGGESTIONS)
                    ]
        return suggestion_row, suggestion_buttons

    def _setup_event_handlers(
        self,
        state: gr.State,
        token_input: gr.Textbox,
        set_token_btn: gr.Button,
        token_row: gr.Row,
        room_selection_row: gr.Row,
        genie_rooms_dropdown: gr.Dropdown,
        chat_row: gr.Row,
        suggestion_row: gr.Row,
        suggestion_buttons: List[gr.Button],
        chatbot: gr.ChatInterface,
    ) -> None:
        """Setup all event handlers for the UI components"""
        # Token submission handler
        set_token_btn.click(
            fn=self.handle_token_submission,
            inputs=[token_input, state],
            outputs=[
                state,
                token_row,
                room_selection_row,
                genie_rooms_dropdown,
            ],
        )

        # Room selection handler
        genie_rooms_dropdown.change(
            fn=self.handle_room_selection,
            inputs=[state, genie_rooms_dropdown],
            outputs=[state, chat_row, suggestion_row] + suggestion_buttons,
        )

        # Suggestion button handlers
        def handle_suggestion(suggestion_text: str) -> str:
            """Handle suggestion button clicks"""
            return suggestion_text

        for btn in suggestion_buttons:
            btn.click(
                fn=handle_suggestion,
                inputs=[btn],
                outputs=[chatbot.textbox],
            ).then(
                fn=None,
                js="""
                () => {
                    // Find and click the submit button
                    document.querySelector('.submit-button').click();
                    return [];
                }
                """,
            )

    def _handle_suggestion_click(self, suggestion_text: str) -> str:
        """Handle when a suggestion button is clicked"""
        return suggestion_text

    def launch(self) -> None:
        """Launch the Gradio interface"""
        self.demo.queue(default_concurrency_limit=self.GRADIO_CONCURRENCY_COUNT)
        self.demo.launch(max_threads=self.GRADIO_MAX_THREADS)


# Create a global instance of the app
app = GenieGradioApp()
demo = app.demo  # Expose the demo instance globally

# Entrypoint
if __name__ == "__main__":
    app.launch()
