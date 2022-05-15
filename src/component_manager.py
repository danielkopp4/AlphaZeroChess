from collections import deque
import json
from os.path import exists, isdir, join
from os import makedirs
from typing import Dict, Optional, Type
import numpy as np
from game import Game
from model import Model
from state import State
from pi import PI
from config_loader import config
from logger import get_logger

logger = get_logger(__name__)

json_file_name = "params.json"

DataSet = Dict[str, deque]

def make_empty_ds(max_len: int):
    ds = {
        "states": deque([], maxlen=max_len),
        "pis": deque([], maxlen=max_len),
        "vs": deque([], maxlen=max_len)
    }
    return ds

class ComponentManager:
    def __init__(self, game: Type[Game], file_path: str, load: bool = False):
        self._nns: Dict[str, Model] = {}

        max_len = config["max_data_points"]
        self._train_split_percent = config["train_percent"]
        self._train_max_len = int(max_len * self._train_split_percent)
        self._test_max_len = max_len - self._train_max_len
        self._train_data: DataSet = make_empty_ds(self._train_max_len)
        self._test_data: DataSet = make_empty_ds(self._test_max_len)

        self._best_nn: Optional[str] = None
        self._file_path: str = file_path
        self._iterations = 0
        self._game = game

        if load and exists(file_path):
            self.load()

    @property
    def game(self) -> Type[Game]:
        return self._game

    def save(self):
        fp = self._file_path
        if exists(fp) and not isdir(fp):
            raise FileExistsError("{} must be a directory".format(fp))

        if not exists(fp):
            makedirs(fp)

        to_save = {
            "nns": list(self._nns.keys()),
            "best_nn": self._best_nn,
            "iterations": self._iterations
        }

        with open(join(fp, json_file_name), "w") as file:
            json.dump(to_save, file)
        # save nns
        for name in self._nns:
            self._nns[name].save(fp, name)

        

    def load(self):
        fp = self._file_path
        if not exists(fp):
            raise FileNotFoundError("tried to load {}, and does not exist".format(fp))
        
        if not isdir(fp):
            raise FileExistsError("{} must be a directory".format(fp))

        with open(join(fp, json_file_name)) as file:
            params = json.load(file)

        nn_names = params["nns"]
        self._best_nn = params["best_nn"]
        self._iterations = params["iterations"]

        self._nns = {}

        for name in nn_names:
            self._nns[name] = self.game.get_model_class().from_file(fp, name)

        # load data set

    def get_best_nn(self) -> Model:
        return self._nns[self._best_nn]

    def add_model(self, name: str, model: Model, best: bool = True):
        self._nns[name] = model
        if best:
            self._best_nn = name

    def del_model(self, name: str):
        del self._nns[name]

    def pop_model(self, name: str) -> Model:
        result = self._nns[name]
        self.del_model(name)
        return result

    def get_model(self, name: str) -> Model:
        return self._nns[name]

    def increment_iter(self):
        self._iterations += 1

    def pop_left_dp(self, train: bool):
        data_set = self.get_data_set(train)

        data_set["states"].popleft()
        data_set["pis"].popleft()
        data_set["vs"].popleft()
        

    def add_data_point(self, state: State, pi: PI, v: float):
        train = bool(np.random.choice([1,0], p=[self._train_split_percent, 1-self._train_split_percent]))

        data_set = self.get_ds(train)

        if 1 + self.ds_length(train) > self.get_max_len(train):
            logger.debug("removing 1 datapoint")
            self.pop_left_dp(train)

        data_set["states"].append(state)
        data_set["pis"].append(pi)
        data_set["vs"].append(v)

    def get_max_len(self, train: bool) -> int:
        if train:
            return self._train_max_len
        else:
            return self._test_max_len

    def get_ds(self, train: bool) -> DataSet:
        if train:
            data_set = self._train_data
        else:
            data_set = self._test_data
        return data_set

    def ds_length(self, train: bool) -> int:
        data_set = self.get_ds(train)

        return len(data_set["states"])

    @property
    def iterations(self):
        return self._iterations

    def get_data_set(self, train: bool) -> Model.DataType:
        data_set = self.get_ds(train)
        return data_set["states"], (data_set["pis"], data_set["vs"])
