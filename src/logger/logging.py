import logging
import sys


def initialise_logger(name: str, overwrite_level=None) -> logging.Logger:
    # TODO: implement overwrite logging level
    logging_level = logging.DEBUG

    logger = logging.getLogger(name)
    # make level configurable
    logger.setLevel(logging_level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s | %(name)s | %(funcName)s"
    )

    # File handler
    file_handler = logging.FileHandler("miniflow.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # prevents duplications
    logger.propagate = False

    return logger
