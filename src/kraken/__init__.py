"""
Placement Engine Package - Modernized operator placement computation

This package provides a clean, modular implementation of the placement engine
that extracts and modernizes the compute_operator_placement_with_prepp functionality.
"""

from .core import compute_operator_placement_with_prepp

__all__ = ['compute_operator_placement_with_prepp']