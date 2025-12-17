import base64
import json
from email.message import EmailMessage

from agent.actions.action import Action
from agent.actions.utils import get_credentials, parse_payload
from agent.controller.state import State
from googleapiclient.discovery import build


class EmailAction(Action):
    def __init__(self, thought: str, payload: str):
        super().__init__(thought, payload)

    def execute(self, state: State) -> str:
        observation = "Email not sent"
        try:
            creds = get_credentials()
            service = build("gmail", "v1", credentials=creds)

            info = parse_payload(self.payload)

            with open("agent/secrets/secrets.json", "r") as f:
                self.email_whitelist = json.load(f)["email_whitelist"]

            if (
                info is not None
                and "to" in info
                and "subject" in info
                and "content" in info
                and info["to"] in self.email_whitelist
            ):

                message = EmailMessage()
                message.set_content(info["content"])
                message["To"] = info["to"]
                message["From"] = "me"
                message["Subject"] = info["subject"]

                encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
                send_message = {"raw": encoded_message}

                # Step 4: Use Gmail API to send
                response = (
                    service.users()
                    .messages()
                    .send(userId="me", body=send_message)
                    .execute()
                )
                print(f'Message ID: {response["id"]}')
                observation = f"Email sent to {info['to']}"
            else:
                observation = "Error: Payload Malformed or Email not in whitelist"
        except Exception as e:
            observation = f"Error: {e}"

        state.history.append(
            {
                "action": {"type": "email", "payload": self.payload},
                "observation": {"type": "email", "payload": observation},
            }
        )
        return observation
