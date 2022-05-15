from dl_agent import DLAgent
from chess_rules.chess_cli_agent import ChessCLIAgent
from game_runner import GameRunner
from chess_rules.chess_game import ChessGame

def run():
    agents = [ChessCLIAgent(), ChessCLIAgent()]
    game_runner = GameRunner(*agents, ChessGame)

    # games_to_play = int(input("How many games? "))
    games_to_play = 1
    assert games_to_play > 0

    game_runner.best_of(games_to_play, 0)

if __name__ == "__main__":
    run()