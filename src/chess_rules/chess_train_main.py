from trainer import Trainer
from config_loader import config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config["tf_log_level"]
from logger import get_logger
from chess_rules.chess_game import ChessGame

logger = get_logger("chess_main_t")

def train():
    logger.info("starting chess trainer...")
    file_path = config["save_path"]
    logger.info("set save location to {}".format(file_path))
    trainer = Trainer(ChessGame, file_path)
    trainer.self_play(config["self_play_iters"])


if __name__ == "__main__":
    train()