from agent.agenthub.base_agent import BaseAgent
from agent.controller.state import State
from agent.actions.action import Action
from agent.llm.openai_chat_completion import get_response
from agent.agenthub.chat_agent.prompts import get_prompt
from agent.actions.chat_action import ChatAction
from agent.actions.calendar_action import CalendarAction
from agent.actions.web_search_action import WebSearchAction
from agent.actions.contact_action import ContactAction
from agent.actions.email_action import EmailAction
import re
import json

class ChatAgent(BaseAgent):
    def __init__(self):
        super().__init__()
       
    def step(self, state: State) -> Action:
        """
        Returns the next action to take based on the current state.
        """ 

        prompt = get_prompt(state.conversation, state.dst, state.history[-1]["action"]["type"])
        response = get_response(prompt)

        thought, action_type, payload = self.parse_response(response)
        print(f"\n*************INTERNAL-MONOLOGUE***************\nThought: {thought}, \nAction Type: {action_type}, \nPayload: {payload} \nConversation History: {state.conversation} \nAction-Observation History: {state.history} \nDST state: {state.dst}\n****************INTERNAL-MONOLOGUE************\n")

        if action_type == "chat":
            action = ChatAction(thought=thought, payload=payload)
        elif action_type == "calendar":
            action = CalendarAction(thought=thought, payload=payload)
        elif action_type == "web_search":
            action = WebSearchAction(thought=thought, payload=payload)
        elif action_type == "contact":
            action = ContactAction(thought=thought, payload=None)
        elif action_type == "email":
            action = EmailAction(thought=thought, payload=payload)
        else:
            action = ChatAction(thought="Invalid action type", payload="There was an error parsing the action type. Please try again.")
        return action
    
    def parse_response(self, response: str) -> tuple[str, str, str]:
        # Action will be within <action> tags
        # Payload will be within <payload> tags
        # Thought will be in <thought>
        thought_match = re.search(r'<thought>\s*(.*?)\s*</thought>', response, re.DOTALL)
        action_match = re.search(r'<action>\s*(.*?)\s*</action>', response, re.DOTALL)
        payload_match = re.search(r'<payload>\s*(.*?)\s*</payload>', response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else "No thought provided"
        action = action_match.group(1).strip() if action_match else "chat"  # Default to chat if no action found
        payload = payload_match.group(1).strip() if payload_match else "No payload provided"

        return thought, action, payload
