"""
Structured logging for the placement engine.

This module provides logging utilities with consistent formatting
and dual ID tracking for original and subgraph node IDs.
"""

import logging
from typing import Optional


def get_kraken_logger(name: str) -> logging.Logger:
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
