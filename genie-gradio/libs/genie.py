import logging
from typing import Optional, List, Dict, Any, Tuple
from functools import partial

from urllib3.util.retry import Retry

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import MessageStatus

import requests
from requests.adapters import HTTPAdapter

from ratelimit import limits, sleep_and_retry


class GenieError(Exception):
    """Base exception for Genie-related errors."""

    pass


class GenieAuthenticationError(GenieError):
    """Raised when authentication fails."""

    pass


class GenieAPIError(GenieError):
    """Raised when the API returns an error."""

    pass


class GenieHandler:
    """
    Handles interactions with Databricks' Genie API.

    This class manages authentication and provides methods to interact with
    Genie rooms, conversations, and queries.

    Attributes:
        databricks_host (str): The Databricks instance hostname
        databricks_user_token (str): Authentication token for Databricks
        logger (logging.Logger): Logger instance for this class
        session (requests.Session): Authenticated session for API requests
        workspace_client (WorkspaceClient): Databricks SDK workspace client
        API_VERSION (str): API version
        ROOMS_ENDPOINT (str): API endpoint for rooms
        GENIE_ENDPOINT (str): API endpoint for Genie spaces
        base_url (str): Base URL for API requests
        DEFAULT_TIMEOUT (int): Default timeout for API requests
        CALLS_PER_SECOND (int): Number of calls allowed per second
        MAX_RETRIES (int): Maximum number of retries for API requests
    """

    # API endpoints
    API_VERSION = "2.0"
    ROOMS_ENDPOINT = "data-rooms"
    GENIE_ENDPOINT = "genie/spaces"

    # Add to class attributes
    DEFAULT_TIMEOUT = 30  # seconds
    CALLS_PER_SECOND = 5
    MAX_RETRIES = 3

    def __init__(self, databricks_host: str, databricks_user_token: str) -> None:
        """
        Initialize the GenieHandler with Databricks credentials.

        Args:
            databricks_host: The Databricks instance hostname
            databricks_user_token: Authentication token for Databricks
        """
        self.logger = logging.getLogger(__name__)
        self.databricks_host = databricks_host
        self.base_url = f"https://{databricks_host}/api/{self.API_VERSION}"
        self.databricks_user_token = databricks_user_token
        self._create_session()

    def _create_session(self) -> None:
        """Initialize authenticated session and workspace client."""
        self.session = requests.Session()

        # Configure retries
        retry_strategy = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.databricks_user_token}",
                "Content-type": "application/json",
                "Accept": "*/*",
            }
        )
        # Set default timeout for all requests
        self.session.request = partial(
            self.session.request, timeout=self.DEFAULT_TIMEOUT
        )

        self.workspace_client = WorkspaceClient(
            host=self.databricks_host, token=self.databricks_user_token
        )

    def get_genie_rooms(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch available Genie rooms.

        Returns:
            List of room dictionaries if successful, None otherwise
        """
        try:
            response = self.session.get(
                url=self._build_url(self.ROOMS_ENDPOINT),
                params={"page_size": 5000},
            )
            data = self._handle_response(response)
            return data.get("data_rooms")

        except (GenieAuthenticationError, GenieAPIError) as e:
            self.logger.error(f"Failed to get Genie rooms: {str(e)}")
            return None

    def get_curated_questions(self, space_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch curated questions for a specific Genie room.

        Args:
            space_id: The ID of the Genie room

        Returns:
            List of curated questions if successful, None otherwise
        """
        try:
            response = self.session.get(
                url=self._build_url(self.ROOMS_ENDPOINT, space_id, "curated-questions"),
                params={"question_type": "SAMPLE_QUESTION"},
            )
            data = self._handle_response(response)
            return data.get("curated_questions")

        except (GenieAuthenticationError, GenieAPIError) as e:
            self.logger.error(
                f"Failed to get curated questions for room {space_id}: {str(e)}"
            )
            return None

    def start_conversation(
        self, space_id: str, content: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Start a new conversation in a Genie room.

        Args:
            space_id: The ID of the Genie room
            content: Initial message content (optional)

        Returns:
            Conversation data if successful, None otherwise
        """
        try:
            response = self.workspace_client.genie.start_conversation_and_wait(
                space_id, content
            )
            if response.status == MessageStatus.COMPLETED:
                return response.as_dict()

            self.logger.error(f"Conversation failed with status: {response.status}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to start conversation: {str(e)}")
            return None

    def create_message(
        self, space_id: str, conversation_id: str, content: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new message in an existing conversation.

        Args:
            space_id: The ID of the Genie room
            conversation_id: The ID of the conversation
            content: Message content

        Returns:
            Message data if successful, None otherwise
        """
        try:
            response = self.workspace_client.genie.create_message_and_wait(
                space_id, conversation_id, content
            )
            if response.status == MessageStatus.COMPLETED:
                return response.as_dict()

            self.logger.error(f"Message creation failed with status: {response.status}")
            return None

        except Exception as e:
            self.logger.error(f"Failed to create message: {str(e)}")
            return None

    def get_message(
        self, space_id: str, conversation_id: str, message_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific message from a conversation.

        Args:
            space_id: The ID of the Genie room
            conversation_id: The ID of the conversation
            message_id: The ID of the message to retrieve

        Returns:
            Message data if successful, None otherwise
        """
        try:
            response = self.session.get(
                url=self._build_url(
                    self.ROOMS_ENDPOINT,
                    space_id,
                    "conversations",
                    conversation_id,
                    "messages",
                    message_id,
                ),
            )
            return self._handle_response(response)

        except (GenieAuthenticationError, GenieAPIError) as e:
            self.logger.error(
                f"Failed to get message {message_id} from conversation {conversation_id}: {str(e)}"
            )
            return None

    def get_query_result(
        self, space_id: str, conversation_id: str, message_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch query results for a specific message.

        Args:
            space_id: The ID of the Genie room
            conversation_id: The ID of the conversation
            message_id: The ID of the message containing the query

        Returns:
            Query result data if successful, None otherwise
        """
        try:
            response = self.session.get(
                url=self._build_url(
                    self.GENIE_ENDPOINT,
                    space_id,
                    "conversations",
                    conversation_id,
                    "messages",
                    message_id,
                    "query-result",
                ),
            )
            return self._handle_response(response)

        except (GenieAuthenticationError, GenieAPIError) as e:
            self.logger.error(
                f"Failed to get query result for message {message_id}: {str(e)}"
            )
            return None

    """
    Helper methods to prettify API responses
    """

    def extract_message_content(self, response: Dict[str, Any]) -> str:
        """
        Extract text content from a message response.

        Args:
            response: The message response dictionary

        Returns:
            Concatenated content from all text attachments
        """
        content_parts = []

        if attachments := response.get("attachments", []):
            for attachment in attachments:
                if text_content := attachment.get("text", {}).get("content"):
                    content_parts.append(text_content)

        return "\n".join(content_parts)

    def transform_query_result(
        self, query_result: Dict[str, Any]
    ) -> Tuple[List[str], List[List[str]]]:
        """
        Transform raw query result into structured column/row format.

        Args:
            query_result: Raw API response containing query result

        Returns:
            Tuple containing:
            - List of column names
            - List of rows (each row is a list of string values)
        """
        try:
            schema = query_result["statement_response"]["manifest"]["schema"]
            columns = [col["name"] for col in schema["columns"]]

            data = query_result["statement_response"]["result"]["data_typed_array"]
            rows = [[value["str"] for value in row["values"]] for row in data]

            return columns, rows

        except KeyError as e:
            self.logger.error(f"Failed to transform query result: {str(e)}")
            return [], []

    def _build_url(self, *parts: str) -> str:
        """
        Build a URL for the Databricks API.

        Args:
            *parts: URL path components to append

        Returns:
            Complete URL string
        """
        return f"{self.base_url}/{'/'.join(part.strip('/') for part in parts)}"

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and raise appropriate exceptions.

        Args:
            response: Response from the API

        Returns:
            Parsed JSON response

        Raises:
            GenieAuthenticationError: If authentication fails
            GenieAPIError: If the API returns an error
        """
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise GenieAuthenticationError("Authentication failed") from e
            raise GenieAPIError(f"API request failed: {str(e)}") from e

    @sleep_and_retry
    @limits(calls=CALLS_PER_SECOND, period=1)
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Make a rate-limited request to the API.

        Args:
            method: HTTP method to use
            url: URL to request
            **kwargs: Additional arguments to pass to requests

        Returns:
            Response from the API
        """
        return self.session.request(method, url, **kwargs)
