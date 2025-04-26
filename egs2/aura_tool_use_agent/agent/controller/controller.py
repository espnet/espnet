from pathlib import Path
wd = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(wd))

from agent.controller.state import State
from agent.agenthub.chat_agent.ChatAgent import ChatAgent
from agent.dst.dst_action import DSTAction
from agent.actions.chat_action import ChatAction


class Controller:
    def __init__(self):
        self.state = State()
        self.agent = ChatAgent()
        self.dst = DSTAction()


    def get_next_chat_action(self):
        action = None
        observation = None
        while not isinstance(action, ChatAction):
            action = self.agent.step(self.state)
            observation = action.execute(self.state)

        return action, observation
    
    def add_user_input(self, user_input):
        self.state.conversation.append({"role": "user", "content": user_input})
        if len(self.state.history) > 0 and self.state.history[-1]['observation']['payload'] is None:
            self.state.history[-1]['observation']['payload'] = user_input
        elif len(self.state.history) == 0:
            self.state.history.append({"action": {"type":None, "payload":None} , 
                              "observation": {"type":"chat", "role":"user", "payload":user_input}})
    
    def reset(self):
        self.state = State()
        self.agent = ChatAgent()
        self.dst = DSTAction()
    
    def run(self):
        while True:
            action = self.agent.step(self.state)
            observation = action.execute()

            if isinstance(action, ChatAction):
                self.state.conversation.extend([{"role": "assistant", "content": action.payload},{"role": "user", "content": observation}])

            self.state.history.extend([{"role": "assistant", "content": action.payload},{"role": "user", "content": observation}])
            

            if self.state.dst_class is None:
                self.state.dst_class = DSTAction(thought='Have to get DST category', payload=state.conversation).execute(type='get_category')['output']
            else:
                self.state.dst = DSTAction(thought='Have to get DST category', payload=self.state.conversation).execute(type=self.state.dst_class)
    



        
            #     self.dst_class = None
            #     #self.history.append({"role": "assistant", "content": initial_dst_classify['response']})
            #     return initial_dst_classify['response']
            # else:
            #     self.dst_class = initial_dst_classify['output']
            #     print("***DST Class set to :", self.dst_class,'***')

    

