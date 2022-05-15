from typing import Dict, Tuple, Type
from game import Game
from state import State
from config_loader import config
from model import Model
import numpy as np
from action import Action
from logger import get_logger
from pi import PI

logger = get_logger(__name__)
EPS = 1e-8

class MCTS:
    CONTINUE = 1000


    def __init__(self, nn: Model, game: Type[Game]):
        self._EV_edge: Dict[Tuple[State, Action], float] = {}
        self._EV_node: Dict[State, float] = {}
        self._N_edge: Dict[Tuple[State, Action], int] = {}
        self._N_node: Dict[State, int] = {}
        self._nn_out: Dict[State, PI] = {}
        self._fin: Dict[State, int] = {}
        self._valid: Dict[State, np.ndarray] = {}
        self._nn: Model = nn
        self._game: Game = game

    def get_pi(self, state: State, tau: float, suppress_warning: bool = False) -> PI:
        for _ in range(config["simulations"]):
            self.update(state, suppress_warning=suppress_warning)

        return self._game.get_PI_class().from_N_temp(self._get_N_mat(state), tau)

    def _get_N_mat(self, state: State) -> np.ndarray:
        action_cls = self._game.get_action_class()
        shape = action_cls.get_shape()
        out = np.zeros(shape)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    action = action_cls.from_action_tuple((i,j,k))
                    if (state, action) not in self._N_edge:
                        out[i][j][k] = 0
                    else:
                        out[i][j][k] = self._N_edge[(state,action)]
        return out

    def update(self, state: State, suppress_warning: bool = False) -> int:
        if state not in self._fin:
            if state.terminal:
                self._fin[state] = state.outcome
            else:
                self._fin[state] = MCTS.CONTINUE

        if self._fin[state] != MCTS.CONTINUE:
            logger.debug("reached terminal with outcome {}".format(self._fin[state]))
            return -self._fin[state]

        if state not in self._nn_out: # not predicted yet, use only nn to find outcome
            pi, v = self._nn.predict_pi_v(state)
            valid_moves = state.legal_action_vector()
            pi = pi.np_arr
            pi_dist = np.multiply(valid_moves, pi)
            pi_sum = np.sum(pi_dist)
            if pi_sum > 0:
                pi_dist /= pi_sum
            else:
                # if not suppress_warning:
                #     logger.warn("pi_sum <= 0, it's ok just should not happen too much, will randomly select move now")
                pi_dist += valid_moves
                pi_dist /= np.sum(pi_dist)

            self._nn_out[state] = self._game.get_PI_class().from_pi_dist(pi_dist)
            self._valid[state] = valid_moves
            self._N_node[state] = 0
            return -v 

        valid_moves = self._valid[state]

        max_u = None
        best_action = None

        n_num = np.sqrt(self._N_node[state])
        n_num_eps = np.sqrt(self._N_node[state] + EPS)
        pred_pi = self._nn_out[state]

        c = config["exploration_coefficient"]

        for action in state.get_legal_actions():
            if (state, action) in self._EV_edge:
                u = self._EV_edge[(state, action)] + c * pred_pi.p_of_a(action) * n_num / (1 + self._N_edge[(state,action)])
            else:
                u = c * pred_pi.p_of_a(action) * n_num_eps
            
            if max_u == None or u > max_u:
                max_u = u
                best_action = action


        next_state = state.take_action(best_action)
        v = self.update(next_state)

        if (state, best_action) in self._EV_edge:
            self._EV_edge[(state, best_action)] = (
                self._N_edge[(state, best_action)] * self._EV_edge[(state, best_action)] + v
            ) / (self._N_edge[(state, best_action)] + 1)

            self._N_edge[(state, best_action)] += 1

        else:
            self._EV_edge[(state, best_action)] = 0
            self._N_edge[(state, best_action)] = 1

        self._N_node[state] += 1
        return -v
