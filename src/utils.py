from typing import Union
from typing import List, Tuple
import numpy as np

def flatten_length(arr: np.ndarray) -> int:
    return np.prod(arr.shape)

def format_multi_dim_index(index: int, shape: tuple) -> tuple:
    out = np.zeros((len(shape),), dtype=np.uint64)
    for i in range(len(shape) - 1, -1, -1):
        k = shape[i]
        out[i] = index % k
        index //= k
    return tuple(out)

def multi_dim_random_choice(pi: np.ndarray) -> tuple:
    return _flatten_approach(pi)

def _flatten_approach(pi: np.ndarray) -> tuple:
    length = flatten_length(pi)
    index_1d = np.random.choice(np.arange(length), p=pi.flatten())
    return format_multi_dim_index(index_1d, pi.shape)

def multi_dim_argmax(arr: np.ndarray) -> tuple:
    index_1d = np.argmax(arr.flatten())
    return format_multi_dim_index(index_1d, arr.shape)

def dot(t1: Tuple[Union[int, float]], t2: Tuple[Union[int, float]]) -> float:
    out = 0
    for a, b, in zip(t1, t2):
        out += a * b
    return out

def left_shift(l1: List[int], l2: List[int]) -> List[int]:
    out = []
    for x, y in zip(l1, l2):
        out.append(x << y)
    return out

def bool_to_nn_int(boolean: bool) -> int:
    return int(boolean) * 2 - 1