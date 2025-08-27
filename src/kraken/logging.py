"""
Structured logging for the placement engine.

This module provides logging utilities with consistent formatting
and dual ID tracking for original and subgraph node IDs.
"""

import logging
from typing import Optional


def get_placement_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for placement engine components.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"placement_engine.{name}")
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
    return logger


def log_dual_id(logger: logging.Logger, level: int, message: str, 
                orig_id: int, sub_id: Optional[int] = None, **kwargs) -> None:
    """
    Log a message with dual ID information (original and subgraph IDs).
    
    Args:
        logger: Logger instance to use
        level: Logging level (e.g., logging.INFO)
        message: Base message to log
        orig_id: Original node ID
        sub_id: Subgraph node ID (optional)
        **kwargs: Additional context to include in the message
    """
    if sub_id is not None:
        id_str = f"node={orig_id} (sub={sub_id})"
    else:
        id_str = f"node={orig_id}"
        
    context_str = ""
    if kwargs:
        context_parts = [f"{k}={v}" for k, v in kwargs.items()]
        context_str = f" {' '.join(context_parts)}"
        
    full_message = f"{message} {id_str}{context_str}"
    logger.log(level, full_message)