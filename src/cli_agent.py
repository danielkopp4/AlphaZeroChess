from abc import ABC, abstractmethod, abstractproperty
from state import State
from action import Action
from agent import Agent

class CLIAgent(Agent, ABC):
    def get_move(self, state: State) -> Action:
        self.display(state)
        input_str = input(self.prompt)
        return self.parse_to_action(input_str, state)

    @abstractmethod
    def parse_to_action(self, input_str: str, state: State) -> Action:
        pass

    @abstractproperty
    def prompt(self) -> str:
        pass

    @abstractmethod
    def display(self, state: State):
        pass