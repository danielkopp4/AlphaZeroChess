from typing import Type
from action import Action
from utils import multi_dim_argmax, multi_dim_argmax, multi_dim_random_choice

from abc import ABC, abstractproperty, abstractstaticmethod
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

class PI(ABC):
    @classmethod
    def from_N_temp(cls, N: np.ndarray, tau: float) -> 'PI':
        # for debugging only, so delete when done
        n_sum = np.sum(N)

        if n_sum <= 0:
            logger.critical("N_SUM <= 0, ERROR WITH MCTS")
            raise Exception("forall i,j,k ==> N[i][j][k] = 0")

        if tau == 0:
            distrib = np.zeros(N.shape)
            distrib[multi_dim_argmax(N)] = 1
        else:
            numerator = np.power(N, 1/tau)
            denominator = np.sum(numerator)

            distrib = numerator / denominator
            logger.debug("tau != 0, denom > 0 ({})".format(denominator))

        # logger.debug("sum of pi dist {}".format(np.sum(distrib)))
        return cls.from_pi_dist(distrib)

    @abstractstaticmethod
    def from_pi_dist(pi_dist: np.ndarray) -> 'PI':
        pass 

    @abstractproperty
    def shape(self) -> tuple:
        pass

    @abstractproperty
    def np_arr(self) -> np.ndarray:
        pass

    def get_rand_action(self, action_cls: Type[Action]) -> Action:
        return action_cls.from_action_tuple(multi_dim_random_choice(self.np_arr))

    def get_best_action(self, action_cls: Type[Action]) -> Action:
        return action_cls.from_action_tuple(multi_dim_argmax(self.np_arr))

    def p_of_a(self, action: Action) -> float:
        return self.np_arr[action.action_tuple]