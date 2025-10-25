import datetime
import logging
from pathlib import Path

# 颜色代码配置
COLOR_CODES = {
    "asctime": "\033[35m",  # Magenta
    "name": "\033[34m",  # Blue
    "filename": "\033[36m",  # Cyan
    "lineno": "\033[34m",
    "funcName": "\033[33m",  # Yellow
    "levelname": {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    },
    "message": "\033[37m",  # White
}
RESET_CODE = "\033[0m"


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        record.name = f"{COLOR_CODES['name']}{record.name}{RESET_CODE}"
        record.filename = f"{COLOR_CODES['filename']}{record.filename}{RESET_CODE}"
        record.funcName = f"{COLOR_CODES['funcName']}{record.funcName}{RESET_CODE}"
        record.levelname = f"{COLOR_CODES['levelname'].get(record.levelname, '')}{record.levelname}{RESET_CODE}"
        record.msg = f"{COLOR_CODES['message']}{record.msg}{RESET_CODE}"
        return super().format(record)


def setup_basic_logging() -> None:
    """Setup basic logging configuration with colored console output"""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # log_format = (
    #     '[%(asctime)s] [%(name)s] {%(filename)s:%(lineno)d} '
    #     '[%(funcName)s] <%(levelname)s> - %(message)s'
    # )
    log_format = (
        "[%(asctime)s] {%(filename)s:%(lineno)d} "
        "[%(funcName)s] <%(levelname)s> - %(message)s"
    )

    # Setup console handler with colored output
    console_formatter = ColoredFormatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Set specific loggers' levels
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)
    logging.getLogger("hydra").setLevel(logging.WARNING)


def setup_file_logging(
    save_dir: Path, mode: str = "train", is_resume: bool = False
) -> None:
    """Setup file logging configuration

    Args:
        save_dir: Directory to save log file
        mode: Either 'train' or 'test'
        is_resume: Whether this is resuming a previous session
    """
    import os

    log_format = (
        "[%(asctime)s] [%(name)s] {%(filename)s:%(lineno)d} "
        "[%(funcName)s] <%(levelname)s> - %(message)s"
    )

    file_formatter = logging.Formatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # 为分布式训练创建不同的日志文件
    rank_suffix = ""
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        rank_suffix = f"_rank{rank}" if rank > 0 else ""
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        rank_suffix = f"_rank{rank}" if rank > 0 else ""

    # Set log filename based on mode and rank
    log_file = save_dir / f"{mode}{rank_suffix}.log"

    # 只有主进程写入会话分隔符
    is_main = rank_suffix == ""
    if is_main:
        # Write session separator
        with open(log_file, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if is_resume:
                f.write(f"Resuming {mode} session at {timestamp}\n")
            else:
                f.write(f"New {mode} session started at {timestamp}\n")
            f.write("=" * 80 + "\n\n")

    file_handler = logging.FileHandler(log_file, mode="a", delay=True)
    file_handler.setFormatter(file_formatter)

    # Add file handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
