import os
import time
from loguru import logger

if os.path.exists("logs") is False:
    os.mkdir("logs")

time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def set_logger():
    logger.remove()
    logger.add(
        f"logs/{time_str}.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <6}</level> | <level>{message}</level>",
        enqueue=True,
        buffering=1,
    )
    return logger
