"""
Placement Engine Package - Modernized operator placement computation

This package provides a clean, modular implementation of the placement engine
that extracts and modernizes the compute_operator_placement_with_prepp functionality.
"""

from .greedy_kraken_core import run_greedy_kraken

__all__ = ["run_greedy_kraken"]
