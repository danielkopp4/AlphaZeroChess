from abc import ABC, abstractproperty, abstractmethod, abstractstaticmethod
from typing import Set
import numpy as np

from action import Action

class State(ABC):
    @abstractstaticmethod
    def from_root_state() -> 'State':
        pass

    @abstractproperty
    def terminal(self) -> bool:
        pass

    @abstractproperty
    def outcome(self) -> int:
        pass

    @abstractmethod
    def legal_action_vector(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_legal_actions(self) -> Set[Action]:
        pass

    @abstractmethod
    def take_action(self, action: Action) -> 'State':
        pass

    @abstractmethod
    def is_legal(self, action: Action) -> bool:
        pass

    @abstractmethod
    def __eq__(self, other: 'State') -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractstaticmethod
    def get_shape() -> tuple:
        pass

    @abstractmethod
    def get_nn_rep(self) -> np.ndarray:
        pass
