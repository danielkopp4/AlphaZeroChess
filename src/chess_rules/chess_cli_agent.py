from chess import Move
from action import Action
from chess_rules.chess_action import ChessAction
from chess_rules.chess_state import ChessState
from cli_agent import CLIAgent
from logger import get_logger

logger = get_logger(__name__)

class ChessCLIAgent(CLIAgent):
    def parse_to_action(self, input_str: str, state: ChessState) -> Action:
        return ChessAction.from_move(Move.from_uci(input_str), state.board.turn)

    @property
    def prompt(self) -> str:
        return "\nEnter move (uci format): "

    def display(self, state: ChessState):
        logger.info("------- Board -------\n" + str(state.board))
        logger.info("---- Legal Moves ---- \n" + str([str(m) for m in state.board.legal_moves]))
        logger.info("---------------------")
        