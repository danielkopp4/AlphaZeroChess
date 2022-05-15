from pi import PI
import numpy as np
from chess_rules.chess_action import ChessAction

class ChessPI(PI):
    def __init__(self, dist: np.ndarray):
        self._dist = dist
        assert self.np_arr.shape == self.shape

    @staticmethod
    def from_pi_dist(pi_dist: np.ndarray) -> PI:
        return ChessPI(pi_dist) 

    @property
    def shape(self) -> tuple:
        return ChessAction.get_shape()

    @property
    def np_arr(self) -> np.ndarray:
        return self._dist