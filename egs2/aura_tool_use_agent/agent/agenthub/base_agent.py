from agent.actions.action import Action
from agent.controller.state import State


class BaseAgent:
    def __init__(self):
        pass

    def step(self, state: State) -> Action:
        pass
