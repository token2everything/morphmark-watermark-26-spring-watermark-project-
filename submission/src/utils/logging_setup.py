"""结构化日志配置"""

import logging
import sys


def setup_logging(level: int = logging.INFO, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger("watermark")
    logger.setLevel(level)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
