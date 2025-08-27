"""
Placement initialization and state setup.

This module handles the initial setup and state preparation needed
for computing projection placement in the network.
"""

import math
from typing import Dict, List, Any
import networkx as nx
from EvaluationPlan import Projection
from helper.filter import getMaximalFilter
from .logging import get_placement_logger

logger = get_placement_logger(__name__)


def initialize_placement_state(combination: List[Any], proj_filter_dict: Dict[str, Any], 
                             no_filter: bool, projection: Any, graph: nx.Graph) -> Dict[str, Any]:
    """
    Prepare and return the initial state needed for computing projection placement in the network.

    This function is responsible for:
      1. **Routing information setup** – Precomputes shortest paths and a routing dictionary
         so later placement logic can quickly check relationships between nodes.
      2. **Filter application** – Determines and attaches maximal filters to certain
         event-type combinations if they are relevant, unless `no_filter` is set.
      3. **Extended combination generation** – Expands the given `combination` with
         any additional event types required by those filters.
      4. **Projection object creation** – Builds a `Projection` object holding the
         projection's identity and its associated filters.

    Args:
        combination (list):
            List of event types (or tuples of event types) that the projection depends on.
        proj_filter_dict (dict):
            Mapping of event-type combinations to their available filters.
        no_filter (bool):
            If True, ignore all filters; if False, apply maximal filters when available.
        projection (str or object):
            Identifier or definition of the projection we are placing.
        graph (networkx.Graph):
            The network topology, where nodes represent processing locations and edges
            represent possible routing paths.

    Returns:
        dict: A dictionary containing the initialized placement state:
            - **routing_dict**: Nested dict describing routing info, including common ancestors.
            - **routing_algo**: Dictionary of all shortest paths between nodes.
            - **filters**: List of `(combination, maximal_filter)` tuples applied.
            - **extended_combination**: The combination plus any added filters (duplicates removed).
            - **projection**: The constructed `Projection` object.
            - **costs**: Initial cost set to `math.inf` (placeholder for later optimization).
            - **best_node**: Initial best node set to `0` (no placement chosen yet).
            - **best_strategy**: Initial strategy placeholder.
    """
    from allPairs import create_routing_dict
    
    logger.debug(f"Initializing placement state for projection: {projection}")
    
    # Create routing structures
    routing_dict = create_routing_dict(graph)
    routing_algo = dict(nx.all_pairs_shortest_path(graph))
    
    logger.debug(f"Created routing structures for {len(graph.nodes)} nodes")

    # Process filters and extend combination
    filters = []
    extended_combination = []

    for proj in combination:
        extended_combination.append(proj)
        if len(proj) > 1 and len(getMaximalFilter(proj_filter_dict, proj, no_filter)) > 0:
            max_filter = getMaximalFilter(proj_filter_dict, proj, no_filter)
            filters.append((proj, max_filter))
            extended_combination.extend(max_filter)
            logger.debug(f"Added filter for projection {proj}: {max_filter}")

    # Remove duplicates from extended combination
    extended_combination = list(set(extended_combination))
    
    logger.debug(f"Extended combination: {len(combination)} -> {len(extended_combination)} items")

    # Create projection object
    my_projection = Projection(projection, {}, [], [], filters)

    placement_state = {
        'routing_dict': routing_dict,
        'routing_algo': routing_algo,
        'filters': filters,
        'extended_combination': extended_combination,
        'projection': my_projection,
        'costs': math.inf,
        'best_node': 0,
        'best_strategy': 'all_push'
    }
    
    logger.info(f"Placement state initialized with {len(filters)} filters, {len(extended_combination)} event types")
    
    return placement_state