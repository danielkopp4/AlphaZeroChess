from typing import Dict, Optional, Tuple, Type
from agent import Agent
import numpy as np
from dl_agent import DLAgent
from game import Game
from logger import get_logger
from model import Model
from state import State

logger = get_logger(__name__)

class GameRunner:
    def __init__(self, agent1: Agent, agent2: Agent, game: Type[Game]):
        self._agent1 = agent1
        self._agent2 = agent2
        self._agents = [agent1, agent2]
        self._game = game

    @staticmethod
    def from_nns(nn1: Model, nn2: Model, game: Type[Game], stochastic: bool = False) -> 'GameRunner':
        dl_a1 = DLAgent(nn1, stochastic, game)
        dl_a2 = DLAgent(nn2, stochastic, game)
        return GameRunner(dl_a1, dl_a2, game)

    # note that wins indicates wins for agent1 and losses indicate losses for agent1
    def best_of(self, N: int, starting_agent: Optional[int] = None) -> Dict[str, int]:
        wtl = [0, 0, 0]

        for i in range(N):
            logger.debug("starting game {} of {}".format(i, N))
            if starting_agent == None: # random starting side
                side = np.random.randint(0, 2)
            else:
                side = starting_agent

            logger.debug("agent {} starting this game".format(side + 1))

            state: State = self._game.get_state_class().from_root_state()

            while not state.terminal:
                agent = self._agents[side]
                action = agent.get_move(state)

                state = state.take_action(action)

                side = (side + 1) % 2

            outcome = state.outcome # 1->white win 

            self._agent1.display_outcome(outcome)
            self._agent2.display_outcome(-outcome)

            logger.debug("game ended with {}".format("white winning" if outcome == 1 else "draw" if outcome == 0 else "black winning"))

            if starting_agent == 1:
                outcome = -outcome

            # outcome: 1->agent 1 wins, -1->agent 2 wins

            logger.debug("game ended with {}".format("agent 1 winning" if outcome == 1 else "draw" if outcome == 0 else "agent 2 winning"))


            wtl[outcome] += 1

        return {
            "wins": wtl[1],
            "ties": wtl[0],
            "losses": wtl[-1],
        }

    @staticmethod 
    def flip_wins(wins: Dict[str, int]) -> Dict[str, int]:
        out = {
            "wins": wins["lossess"],
            "ties": wins["ties"],
            "losses": wins["wins"]
        }
        return out

    @staticmethod
    def dict_to_win_frac(wins: Dict[str, int]) -> float:
        return GameRunner.wtl_to_win_frac((wins["wins"], wins["ties"], wins["losses"]))

    @staticmethod
    def wtl_to_win_frac(wtl: tuple) -> float:
        return sum(np.multiply(wtl, (1, 0, -1))) / sum(wtl)