from abc import ABC, abstractmethod, abstractproperty, abstractstaticmethod
from typing import Tuple

ActionTuple = Tuple[int, int, int]

class Action(ABC):
    @abstractmethod
    def __init__(self, action_tuple: ActionTuple):
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: 'Action') -> bool:
        pass

    @abstractproperty
    def is_valid(self) -> bool:
        pass

    @abstractstaticmethod
    def get_shape() -> Tuple[int, int, int]:
        pass

    @abstractproperty
    def action_tuple(self) -> ActionTuple:
        pass

    @abstractstaticmethod
    def from_action_tuple(action_tuple: ActionTuple) -> 'Action':
        pass
