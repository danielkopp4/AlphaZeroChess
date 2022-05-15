from abc import ABC, abstractproperty, abstractmethod
from typing import List, Tuple
from state import State
from pi import PI
import numpy as np
from os.path import join
from logger import get_logger

logger = get_logger(__name__)

TestTrainSplit = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
XY = Tuple[np.ndarray, np.ndarray]

class Model(ABC):
    DataType = Tuple[List[State], Tuple[List[PI], List[float]]]

    @classmethod
    def from_file(cls, file_path: str, name: str):
        new_model = cls()
        new_model.load(file_path, name)
        return new_model

    @abstractmethod
    def predict_pi_v(self, state: State) -> Tuple[PI, float]:
        pass

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def save(self, file_path: str, name: str):
        pass

    @abstractmethod
    def load(self, file_path: str, name: str):
        pass

    @abstractmethod
    def clone(self) -> 'Model':
        pass

    @abstractmethod
    def train(self, data: DataType):
        pass

    def get_file_name(self, model_name: str) -> str:
        return self.name + "_" + model_name 

    def get_full_filepath(self, file_path: str, model_name: str) -> str:
        return join(file_path, self.get_file_name(model_name))

    def data_to_np(self, data: DataType, shuffle: bool = True) -> XY:
        states, (pis, vs) = data
        vs = np.array(vs)
        pis = np.array(pis)
        states = np.array(states)

        logger.debug("shapes: {} {} {}".format(vs.shape, pis.shape, states.shape))

        logger.debug("length {}".format(len(states)))


        if shuffle:
            perm = np.random.permutation(len(states))
        else:
            perm = np.arange(len(states))

        states = np.array([state.get_nn_rep() for state in states[perm]])
        pis = np.array([pi.np_arr for pi in pis[perm]])
        vs = np.array(vs[perm]).reshape(len(vs), 1)

        logger.debug("shapes after shuffle and processing: {} {} {}".format(vs.shape, pis.shape, states.shape))

        return states, (pis, vs)

