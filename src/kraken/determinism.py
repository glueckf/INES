"""
Determinism utilities for reproducible placement computations.

This module provides utilities to ensure deterministic behavior across
placement engine executions, including sorting functions and ID generation.
"""

import os
from typing import List, Any, Iterable


def ensure_deterministic_iteration(items: Iterable[Any], key_func=None) -> List[Any]:
    """
    Ensure deterministic iteration order by sorting items.

    Args:
        items: Iterable to make deterministic
        key_func: Optional key function for sorting

    Returns:
        Sorted list for deterministic iteration
    """
    if key_func:
        return sorted(items, key=key_func)
    else:
        # Try to sort directly, fallback to string representation
        try:
            return sorted(items)
        except TypeError:
            return sorted(items, key=str)


def setup_deterministic_environment() -> None:
    """
    Set up environment variables for deterministic execution.

    This sets PYTHONHASHSEED=0 if not already set, which is critical
    for deterministic dict ordering and hash-based operations.
    """
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "0"
        # Note: This only affects subprocesses, not the current process


def validate_deterministic_inputs(candidates: List[Any], logger) -> List[Any]:
    """
    Validate and sort candidate list for deterministic processing.

    This function ensures that candidate processing order is deterministic
    regardless of the input order or underlying data structure ordering.

    Args:
        candidates: List of candidate nodes/items
        logger: Logger instance for warnings

    Returns:
        Sorted list of candidates for deterministic processing
    """
    if not candidates:
        logger.warning("Empty candidate list provided")
        return []

    try:
        # Sort candidates to ensure deterministic order
        sorted_candidates = ensure_deterministic_iteration(candidates)

        if sorted_candidates != candidates:
            logger.debug(
                f"Candidate order changed for determinism: {candidates} -> {sorted_candidates}"
            )

        return sorted_candidates

    except Exception as e:
        logger.warning(f"Could not sort candidates deterministically: {e}")
        return candidates
