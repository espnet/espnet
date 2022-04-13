from typing import Type
from text_preparation.flow.mutators.base import AbstractMutator


class FlowExecutor(object):

    def __init__(self, *args: Type[AbstractMutator]):
        self.dependencies = set()
        self.instances = list()

        for rule in args:
            diff = rule.get_dependencies().difference(self.dependencies)
            if len(diff) > 0:
                raise ValueError('The rule %s expects dependencies: %s' % (rule.__name__, ' '.join(diff)))

            self.dependencies.add(rule.__name__)
            self.instances.append(rule())

    def run(self, text: str) -> str:
        for rule in self.instances:
            text = rule(text)

        return text
