from sys import argv
from typing import Type
from threading import Thread
from tqdm import tqdm
from component_manager import ComponentManager
from game import Game
from config_loader import config
from logger import get_logger
from model import Model
from mcts import MCTS
from pi import PI
from game_runner import GameRunner

logger = get_logger(__name__)

class Trainer:
    def __init__(self, game: Type[Game], file_path: str, load: bool = config["load_prev"]):
        self._component_manager = ComponentManager(game, file_path, load)
        self._component_manager.add_model("base_nn", game.get_model_class()())

    def _train_thread(self, nn: Model, game: Type[Game], first_iter: bool):
        mcts = MCTS(nn, game)
        new_data = self.run_game(mcts, first_iter)
        for state, pi, v in new_data:
            self._component_manager.add_data_point(state, pi, v)

    def self_play(self, iters: int):
        start_iter = self._component_manager.iterations
        final_iter = start_iter + iters

        for _ in range(iters):
            logger.info("starting iteration {}, out of {}".format(self._component_manager.iterations, final_iter))

            first_iter = self._component_manager.iterations == 0

            nn = self._component_manager.get_model("base_nn")
            game = self._component_manager.game

            mcts_threads = [Thread(target=self._train_thread, args=(nn, game, first_iter), daemon=True) for _ in range(config["episodes"])]

            for t in mcts_threads:
                t.start()

            for t in tqdm(mcts_threads, desc="episodes"):
                t.join()

            logger.info("finished self play, training now...")
                

            prev_nn = nn.clone()
            nn.train(
                self._component_manager.get_data_set(True), 
                self._component_manager.get_data_set(False)
            )

            logger.info("comparing prev nn with new")

            game_runner: GameRunner = GameRunner.from_nns(nn, prev_nn, self._component_manager.game)

            if config["side_advantage"]:
                side_1_dict = game_runner.best_of(config["competition_amount"], 0)
                side_1 = GameRunner.dict_to_win_frac(side_1_dict)

                side_2_dict = game_runner.best_of(config["competition_amount"], 1)
                side_2 = GameRunner.dict_to_win_frac(side_2_dict)

                if side_1 > side_2:
                    logger.info("new nn is better {}".format(side_1_dict))
                elif side_2 > side_1:
                    logger.info("old nn was better {}".format(side_2_dict))
                else:
                    logger.info("neither nn was better")

            else:
                win_dict = game_runner.best_of(config["competition_amount"])
                win_frac = GameRunner.dict_to_win_frac(win_dict)
                if win_frac == 0:
                    logger.info("neither nn was better")
                elif win_frac > 0.5:
                    logger.info("new nn is better {}".format(win_dict))
                else:
                    logger.info("old nn was better {} (inverted: losses -> old nn wins)".format(win_dict))


            self._component_manager.increment_iter()
            self._component_manager.save()


    def run_game(self, mcts: MCTS, first_iter: bool = False):
        state = self._component_manager.game.get_state_class().from_root_state()
        step_count = 0

        states = []
        pis = []

        while not state.terminal:

            step_count += 1

            tau = int(step_count < config["tau_threshold"])
            pi: PI = mcts.get_pi(state, tau, first_iter)

            states.append(state)
            pis.append(pi)

            action = pi.get_rand_action(self._component_manager.game.get_action_class())
            state = state.take_action(action) # state containts player

        vs = []

        v = state.outcome
        for _ in range(len(states)):
            vs.append(v)
            v *= -1

        return zip(states, pis, vs)
                