"""
Structured logging for the placement engine.

This module provides logging utilities with consistent formatting
and dual ID tracking for original and subgraph node IDs.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def get_kraken_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Get a configured logger for placement engine components.

    Args:
        name: Logger name (typically __name__ from calling module)
        log_file: Optional file path to save logs to. If None, only console output.
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"kraken.{name}")

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger


def setup_backtracking_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Setup a specialized logger for backtracking algorithm with file output.

    Args:
        log_dir: Directory to save log files

    Returns:
        Configured logger for backtracking operations
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"backtracking_kraken_{timestamp}.log")

    # Create the logger with both console and file output
    logger = get_kraken_logger(
        "backtracking_core", log_file=log_file, level=logging.DEBUG
    )

    # Log the file location for reference
    logger.info(f"Backtracking algorithm log saved to: {os.path.abspath(log_file)}")

    return logger
