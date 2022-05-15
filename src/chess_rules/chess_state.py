from collections import deque
from state import State
from action import Action
from chess_rules.chess_action import ChessAction
from typing import Dict, Optional, Set
import numpy as np
from chess import BLACK, PAWN, PIECE_TYPES, QUEEN, WHITE, Board, Move
from utils import left_shift, bool_to_nn_int
from threading import Lock

class HashableBoard(Board):
    def __hash__(self):
        b_boards = (self.occupied, self.pawns, self.bishops, self.kings, self.queens, self.rooks, self.knights, self.fullmove_number, self.halfmove_clock)
        return sum(left_shift(
            b_boards,
            list(range((len(b_boards)-1) * 64, -1, -64))
        ))

def bit_board_to_arr(bitboard: int, color: bool) -> np.ndarray:
    out = np.zeros((8,8))

    out = np.array([int(i) for i in list('{:064b}'.format(bitboard))]).reshape(8,8)

    # its flipped lr (x) but idt that matters

    if color == BLACK: # flip y
        out = np.flipud(out)

    return out


state_map: Dict[HashableBoard, 'ChessState'] = {}
max_states = 10000 # about 6.4 gb
state_stores = deque([], maxlen=max_states)


cache_lock = Lock()


class ChessState(State):
    def __init__(self, board: HashableBoard):
        self._board: HashableBoard = board
        self._legal_actions: Optional[Set[Action]] = None
        self.get_legal_actions()
        self._legal_actions_vector_set: bool = False
        self._legal_actions_vector: Optional[np.ndarray] = None
        self.legal_action_vector()
        self._nn_rep_set: bool = False
        self._nn_rep: Optional[np.ndarray] = None
        self.get_nn_rep()

    @staticmethod
    def create_state(board: HashableBoard = HashableBoard()) -> 'ChessState':
        if board in state_map:
            return state_map[board]

        cache_lock.aquire()
        if len(state_stores) + 1 >= max_states:
            b = state_stores.popleft()
            del state_map[b]
        cache_lock.release()

        new_chess_state = ChessState(board)
        state_map[board] = new_chess_state
        state_stores.append(board)
        return new_chess_state

    @staticmethod
    def from_root_state() -> State:
        return ChessState.create_state()

    @property
    def board(self) -> Board:
        return self._board

    @property
    def terminal(self) -> bool:
        return self.board.is_game_over()

    @property
    def outcome(self) -> int:
        result = self.board.outcome().result()
        if result == "1/2-1/2":
            return 0

        return 1 if result == "1-0" else -1 # check if this is correct

    def legal_action_vector(self) -> np.ndarray:
        if self._legal_actions_vector_set:
            return self._legal_actions_vector

        self._legal_actions_vector = np.zeros(ChessAction.get_shape(), dtype=np.float16)
        legal_actions = self.get_legal_actions()
        for action in legal_actions:
            self._legal_actions_vector[action.action_tuple] = 1

        self._legal_actions_vector_set = True
        return self._legal_actions_vector


    @staticmethod
    def get_shape() -> tuple:
        return (8, 8, 19) # might not be the right size, not going to include three fold repetition (paper )

    def get_legal_actions(self) -> Set[Action]:
        if self._legal_actions != None:
            return self._legal_actions

        self._legal_actions = set()

        for move in self.board.legal_moves:
            self._legal_actions.add(ChessAction.from_move(move, self.board.turn))

        return self._legal_actions

    def take_action(self, action: ChessAction) -> State:
        new_board: Board = self.board.copy() # if stack=False it wont check for threefold repetition, but expensive if true
        move = action.to_move(new_board.turn)

        # autopromote to queen if possible
        if new_board.piece_at(move.from_square) == PAWN and (
            (new_board.color_at(move.from_square) == WHITE and move.to_square // 8 == 0) 
         or (new_board.color_at(move.from_square) == BLACK and move.to_square // 8 == 7)):
            move = Move(move.from_square, move.to_square, promotion=QUEEN)
        
        new_board.push(move)

        new_state = ChessState.create_state(new_board)
        new_state._board = new_board
        # new_state.board.move_stack = new_board.move_stack # because we save the old states we must copy the move stack
        return new_state

    def is_legal(self, action: ChessAction) -> bool:
        return action in self.get_legal_actions()

    def __eq__(self, other: State) -> bool:
        if other is self:
            return True

        if type(other) != ChessState:
            return False

        if hash(self) != hash(other):
            return False

        other: ChessState = other
        return self.board == other._board

    def __hash__(self) -> int:
       return hash(self.board)

    def get_nn_rep(self) -> np.ndarray:
        if self._nn_rep_set:
           return self._nn_rep

        self._nn_rep = np.zeros(tuple(reversed(ChessState.get_shape())))

        i = 0

        current_color = self.board.turn
        opp_color = not current_color

        for color in [current_color, opp_color]:
            for piece_type in PIECE_TYPES:
                self._nn_rep[i] = bit_board_to_arr(self.board.pieces_mask(piece_type, color), color)
                i += 1

        self._nn_rep[i] = bool_to_nn_int(current_color)
        i += 1
        self._nn_rep[i] = (self.board.fullmove_number / 20)
        i += 1
        self._nn_rep[i] = bool_to_nn_int(self.board.has_queenside_castling_rights(current_color))
        i += 1
        self._nn_rep[i] = bool_to_nn_int(self.board.has_kingside_castling_rights(current_color))
        i += 1
        self._nn_rep[i] = bool_to_nn_int(self.board.has_queenside_castling_rights(opp_color))
        i += 1
        self._nn_rep[i] = bool_to_nn_int(self.board.has_kingside_castling_rights(opp_color))
        i += 1
        self._nn_rep[i] = (self.board.halfmove_clock / 10)

        self._nn_rep = np.transpose(self._nn_rep, (2, 1, 0))

        self._nn_rep_set = True
        return self._nn_rep
