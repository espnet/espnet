from agent.controller.state import State
from agent.actions.action import Action

class BaseAgent:
    def __init__(self):
        pass

    def step(self, state: State) -> Action:
        pass
