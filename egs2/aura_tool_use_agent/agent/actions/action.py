from agent.controller.state import State

class Action:
    def __init__(self, thought: str = '', payload: str = ''):
        self.thought = thought
        self.payload = payload
        pass


    def execute(self, state: State):
        pass


        
