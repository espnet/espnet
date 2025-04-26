from agent.actions.action import Action
from agent.controller.state import State

class ChatAction(Action):
    def __init__(self, thought: str, payload: str):
        super().__init__(thought, payload)

    def execute(self, state: State) -> str:
        state.conversation.append({"role": "assistant", "content": self.payload})
        state.history.append({"action": {"type":"chat", "role":"assistant", "payload":self.payload} , 
                              "observation": {"type":"chat", "role":"user", "payload":None}})

        return None