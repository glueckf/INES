"""
Hardcoded Topology Module

This module provides hardcoded network topologies and scenarios for 
reproducible INES simulations and testing.
"""

from .hardcoded_scenario import (
    create_hardcoded_network,
    create_hardcoded_queries,
    get_hardcoded_eventrates,
    get_hardcoded_primitive_events,
    get_hardcoded_parameters,
    get_expected_results,
    print_hardcoded_scenario_info
)

__all__ = [
    'create_hardcoded_network',
    'create_hardcoded_queries', 
    'get_hardcoded_eventrates',
    'get_hardcoded_primitive_events',
    'get_hardcoded_parameters',
    'get_expected_results',
    'print_hardcoded_scenario_info'
]