from typing import Type
from action import Action
from agent import Agent
from game import Game
from mcts import MCTS
from model import Model
from state import State

class DLAgent(Agent):
    def __init__(self, nn: Model, stochastic: bool, game: Type[Game]):
        self._nn = nn
        self._stochastic = stochastic
        self._mcts = MCTS(nn, game)
        self._game = game

    def get_move(self, state: State) -> Action:
        pi = self._mcts.get_pi(state, int(self._stochastic))

        if self._stochastic:
            return pi.get_rand_action(self._game.get_action_class())

        return pi.get_best_action(self._game.get_action_class())

    def display_outcome(self, outcome: int):
        self._mcts = MCTS(self._nn, self._game)