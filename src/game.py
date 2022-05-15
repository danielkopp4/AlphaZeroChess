from abc import ABC, abstractstaticmethod
from state import State
from action import Action
from pi import PI
from model import Model
from typing import Type

class Game(ABC):
    @abstractstaticmethod
    def get_state_class() -> Type[State]:
        pass

    @abstractstaticmethod
    def get_action_class() -> Type[Action]:
        pass

    @abstractstaticmethod
    def get_model_class() -> Type[Model]:
        pass

    @abstractstaticmethod
    def get_PI_class() -> Type[PI]:
        pass
