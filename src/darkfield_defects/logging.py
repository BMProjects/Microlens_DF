"""统一日志模块 — 全项目使用 get_logger() 替代 print()."""

import logging
import sys
from typing import Optional


_CONFIGURED = False
_BASE_LOGGER_NAME = "darkfield_defects"


def get_logger(
    name: str = "darkfield_defects",
    level: Optional[int] = None,
) -> logging.Logger:
    """获取统一配置的日志器.

    Args:
        name: 日志器名称，默认为包名.
        level: 日志级别，默认 INFO.

    Returns:
        配置好的 Logger 实例.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        if level is None:
            level = logging.INFO

        # 统一将 handler 挂到包级父 logger，子 logger 走传播输出
        base_logger = logging.getLogger(_BASE_LOGGER_NAME)
        base_logger.setLevel(level)
        base_logger.propagate = False

        if not base_logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(level)
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)

        _CONFIGURED = True

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
