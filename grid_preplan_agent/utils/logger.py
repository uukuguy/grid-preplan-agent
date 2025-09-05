import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def setup_logger(
    name: str = "grid_agent",
    level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 可选的日志文件路径
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 如果已经有处理器，先清除
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建Rich处理器用于美化控制台输出
    console = Console()
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=True
    )
    rich_handler.setFormatter(logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    ))
    logger.addHandler(rich_handler)
    
    # 可选的文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger


# 全局默认日志记录器
logger = setup_logger()