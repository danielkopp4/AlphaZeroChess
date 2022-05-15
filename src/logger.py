import logging
import coloredlogs
from config_loader import config

formatting = "[{asctime},{msecs:03.0f}] {name:>15s}:{levelname:<12s} {message}"

logging.basicConfig(
    filename="test_log.log", 
    filemode="w",
    format=formatting,
    style='{'
)

def get_logger(name: str, level: str = config["log_level"]) -> logging.Logger:
    last_dot = name.rfind(".")
    name = name[last_dot+1:]

    logger = logging.getLogger(name)
    coloredlogs.install(
        level=level, 
        fmt=formatting, 
        style="{", 
        logger=logger
    )

    logger.debug("logger '{}' loaded".format(name))

    return logger
    