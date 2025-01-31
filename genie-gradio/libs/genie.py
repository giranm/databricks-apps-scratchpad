import json
import logging

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import MessageStatus

import requests

from prettytable import PrettyTable


# Function to convert raw JSON string to a native dict
def convert_json_string_to_dict(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        return str(e)


class GenieHandler:
    def __init__(self, databricks_host, databricks_user_token):
        self.logger = logging.getLogger(__name__)
        self.databricks_host = databricks_host
        self.databricks_user_token = databricks_user_token
        self.__create_session()

    def __create_session(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.databricks_user_token}",
                "Content-type": "application/json",
                "Accept": "*/*",
            }
        )
        self.workspace_client = WorkspaceClient(
            host=self.databricks_host, token=self.databricks_user_token
        )

    def get_genie_rooms(self):
        res = self.session.get(
            url=f"https://{self.databricks_host}/api/2.0/data-rooms?page_size=5000",
        )

        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(
                f"Failed to get Genie rooms: {res.status_code} - {res.text}"
            )
            return None

    def get_curated_questions(self, space_id):
        res = self.session.get(
            url=f"https://{self.databricks_host}/api/2.0/data-rooms/{space_id}/curated-questions?question_type=SAMPLE_QUESTION",
        )
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(
                f"Failed to get curated questions: {res.status_code} - {res.text}"
            )
            return None

    def start_conversation(self, space_id, content=""):
        res = self.workspace_client.genie.start_conversation_and_wait(space_id, content)
        if res.status == MessageStatus.COMPLETED:
            return res.as_dict()
        else:
            self.logger.error(f"Failed to start conversation: {res.status}")
            return None

    def create_message(self, space_id, conversation_id, content):
        res = self.workspace_client.genie.create_message_and_wait(
            space_id, conversation_id, content
        )
        if res.status == MessageStatus.COMPLETED:
            return res.as_dict()
        else:
            self.logger.error(f"Failed to create message: {res.status}")
            return None

    def get_message(self, space_id, conversation_id, message_id):
        res = self.session.get(
            url=f"https://{self.databricks_host}/api/2.0/data-rooms/{space_id}/conversations/{conversation_id}/messages/{message_id}",
        )
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to get message: {res.status_code} - {res.text}")
            return None

    def get_query_result(self, space_id, conversation_id, message_id):
        res = self.session.get(
            url=f"https://{self.databricks_host}/api/2.0/data-rooms/{space_id}/conversations/{conversation_id}/messages/{message_id}/get-query-result",
        )
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(
                f"Failed to get query result: {res.status_code} - {res.text}"
            )
            return None

    """
    Helper methods to prettify API responses
    """

    def prettify_query_result(self, query_result):
        result_raw_json, manifest_raw_json = (
            query_result["result"]["result_raw_json"],
            query_result["result"]["manifest_raw_json"],
        )
        result_converted_json, manifest_converted_json = map(
            convert_json_string_to_dict, [result_raw_json, manifest_raw_json]
        )

        column_header_names = [
            column["name"] for column in manifest_converted_json["schema"]["columns"]
        ]
        rows_str_extracted = [
            [item.get("str", None) for item in entry["values"]]
            for entry in result_converted_json[
                "data_typed_array"
            ]  # TODO handle empty table
        ]

        table_as_str = PrettyTable()
        table_as_str.field_names = column_header_names
        table_as_str.add_rows(rows=rows_str_extracted)
        return table_as_str

    def extract_message_content(self, response):
        content = ""
        if response.get("attachments") and len(response["attachments"]) > 0:
            for attachment in response["attachments"]:
                if attachment.get("text").get("content"):
                    content += attachment["text"]["content"] + "\n"

        return content
