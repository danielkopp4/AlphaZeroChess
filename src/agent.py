from action import Action
from state import State

from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def get_move(self, state: State) -> Action:
        pass

    def display_outcome(self, outcome: int):
        pass