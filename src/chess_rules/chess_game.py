from chess_rules.chess_action import ChessAction
from chess_rules.chess_model import ChessModel
from chess_rules.chess_pi import ChessPI
from chess_rules.chess_state import ChessState
from game import Game
from state import State
from action import Action
from pi import PI
from model import Model
from typing import Type

class ChessGame(Game):
    @staticmethod
    def get_state_class() -> Type[State]:
        return ChessState

    @staticmethod
    def get_action_class() -> Type[Action]:
        return ChessAction

    @staticmethod
    def get_model_class() -> Type[Model]:
        return ChessModel

    @staticmethod
    def get_PI_class() -> Type[PI]:
        return ChessPI
