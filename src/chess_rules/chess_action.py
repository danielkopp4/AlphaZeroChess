from collections import deque
from typing import Dict, Tuple
from action import Action, ActionTuple
import numpy as np
from chess import Move, QUEEN, Color, BLACK
from chess_rules.chess_square import ChessSquare
from logger import get_logger
from threading import Lock

cache_lock = Lock()

logger = get_logger(__name__)

queen_directions = [
    (0, 1), # north
    (1, 0), # east
    (0, -1), # south
    (-1, 0), # west
    (1, 1), # north east
    (1, -1), # south east
    (-1, -1), # south west
    (-1, 1) # north west
]

for i, direction in enumerate(queen_directions):
    queen_directions[i] = ChessSquare(*direction)

knight_directions = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2)
]

for i, direction in enumerate(knight_directions):
    knight_directions[i] = ChessSquare(*direction)

underpromotion_moves = [ # 3 for each direction
    (0, 1),
    (-1, 1),
    (1, 1)
] # can promote to queen of none are set

for i, direction in enumerate(underpromotion_moves):
    underpromotion_moves[i] = ChessSquare(*direction)

action_map: Dict[ActionTuple, 'ChessAction'] = {}
max_actions = 1000000
action_stores = deque([], maxlen=max_actions)

eps = 1E-8

class ChessAction:
    def __init__(self, action_tuple: ActionTuple, from_creator=False):
        if not from_creator:
            logger.critical("NOT CREATED FROM CREATOR!")
            assert False
        self._action_tuple = action_tuple
        self.parse_move()

    @staticmethod
    def from_action_tuple(action_tuple: ActionTuple) -> 'ChessAction':     
        action_tuple = tuple(int(x) for x in action_tuple)
        if action_tuple in action_map:
            return action_map[action_tuple]

        cache_lock.aquire()
        if len(action_stores) + 1 >= max_actions:
            at = action_stores.popleft()
            del action_map[at]
        cache_lock.release()

        new_chess_action = ChessAction(action_tuple, from_creator=True)
        action_map[action_tuple] = new_chess_action
        action_stores.append(action_tuple)
        return new_chess_action

    def parse_move(self):
        logger.debug("parsing move {}".format(self.action_tuple))
        x, y, p = self.action_tuple
        self._action_type = p
        self._starting_square: ChessSquare = ChessSquare(x, y)
        logger.debug("starting square is {}".format(self._starting_square))
        self._move_type = self._get_move_type()
        logger.debug("move type is {}".format(self._move_type))
        self._to_square: ChessSquare = self._get_to_square()
        logger.debug("ending square {}".format(self._to_square))

    def _get_direction(self) -> int:
        if self._move_type == "queen":
            return self._action_type // 7
        elif self._move_type == "knight":
            return self._action_type - 56

        return (self._action_type - 64) // 3

    def _get_promoted_piece(self) -> int:
        if self._move_type != "underpromotions":
            return None

        return (self._action_type - 64) % 3 + 2 # knight, bishop, rook = [2,3,4] 

    def _get_num_moves(self) -> int:
        assert self._move_type == "queen"
        return 1 + (self._action_type % 7)

    def _get_move_type(self) -> str:
        if self._action_type < 56:
            return 'queen'
        if self._action_type < 64:
            return 'knight'

        return 'underpromotions'

    def _get_to_square(self) -> ChessSquare:
        direction = self._get_direction()
        logger.debug("direction is {}".format(direction))
        logger.debug("direction typpe is {}".format(type(direction)))

        if self._move_type == "queen":
            move_amount = queen_directions[direction]
            logger.debug("queen move dir {}".format(move_amount))
            logger.debug("queen move amount {}".format(self._get_num_moves()))
            move_amount *= self._get_num_moves()
        elif self._move_type == "knight":
            move_amount = knight_directions[direction]
        else:
            move_amount = underpromotion_moves[direction]
            
        return move_amount + self._starting_square

    @staticmethod
    def from_move(move: Move, turn: Color) -> 'ChessAction':
        x = y = p = 0

        logger.debug("creating action from move {}".format(str(move)))

        from_square = ChessSquare.from_CSquare(move.from_square)
        logger.debug("start square: {}".format(from_square))

        to_square = ChessSquare.from_CSquare(move.to_square)
        logger.debug("end square: {}".format(to_square))

        if turn == BLACK:
            logger.debug("turn is black so flipping...")
            logger.debug("start before flip {}, after {}".format(from_square, from_square.color_flip()))
            from_square = from_square.color_flip()
            logger.debug("end before flip {}, after flip {}".format(to_square, to_square.color_flip()))
            to_square = to_square.color_flip()

        diff = to_square - from_square

        logger.debug("move has diff of {}".format(diff))

        x = from_square.x
        y = from_square.y

        if move.promotion == None or move.promotion == QUEEN:
            slope = abs(diff).slope()

            logger.debug("move slope is {}".format(slope))

            if abs(slope - 0.5) < eps or abs(slope - 2) < eps:
                # knight move
                logger.debug("move is knight")
                direction = knight_directions.index(diff)
                p += 56 + direction
            else:
                # queen move
                logger.debug("move is queen")
                diff_a = abs(diff)
                move_amount = max(diff_a.x, diff_a.y)
                diff //= move_amount
                direction = queen_directions.index(diff)
                p += move_amount + direction * 7 - 1
        else:
            p += 64

            logger.debug("move is an underpromotion")
            move_dir = underpromotion_moves.index(diff)
            under_prom_type = move.promotion - 2

            p += 3 * move_dir + under_prom_type

        x = int(x)
        y = int(y)
        p = int(p)

        logger.debug("creating action with tuple: {}".format((x,y,p)))

        return ChessAction.from_action_tuple((x, y, p))

    def to_move(self, turn: Color) -> Move:
        if not self.is_valid():
            logger.critical("action 'invalid' with tuple {} which is from {} to {}".format(self.action_tuple, self._starting_square, self._to_square))
            return None

        flipped_start: ChessSquare = self._starting_square
        flipped_end: ChessSquare = self._to_square

        if turn == BLACK:
            logger.debug("turn is black so flipping...")
            logger.debug("start before flip {}, after {}".format(flipped_start, flipped_start.color_flip()))
            flipped_start = flipped_start.color_flip()
            logger.debug("end before flip {}, after flip {}".format(flipped_end, flipped_end.color_flip()))
            flipped_end = flipped_end.color_flip()


        return Move(flipped_start.chess_square, flipped_end.chess_square, promotion=self._get_promoted_piece()) # get promotion 

    def __hash__(self) -> int:
        return int(np.dot(self.action_tuple, (1, 8, 64)))

    def __eq__(self, other: Action) -> bool:
        if self is other:
            return True

        if type(other) != ChessAction:
            return False
            
        return self.action_tuple == other.action_tuple

    def is_valid(self) -> bool:
        return not self._starting_square.invalid and not self._to_square.invalid

    @staticmethod
    def get_shape() -> Tuple[int, int, int]:
        return (8, 8, 73)

    @property
    def action_tuple(self) -> ActionTuple:
        return self._action_tuple