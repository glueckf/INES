from functools import lru_cache
from typing import Dict, List, Any, Set

import networkx as nx

from src.kraken.candidate_selection import check_possible_placement_nodes_for_input


class PlacementOptimizer:
    """
    Optimized placement logic with caching and reduced redundant computations.
    """

    def __init__(self, graph, routing_dict: Dict[int, Dict[int, Dict[str, Any]]]):
        # Handle both dict and NetworkX graph inputs
        if isinstance(graph, dict):
            self.graph = graph
            self._nx_graph = None
        else:
            # Assume it's already a NetworkX graph
            self._nx_graph = graph
            self.graph = None

        self.routing_dict = routing_dict
        self._reachability_cache = {}  # Cache for reachability computations
        self._ancestor_cache = {}  # Cache for ancestor computations

    @property
    def nx_graph(self):
        """Lazy NetworkX graph creation or return existing graph"""
        if self._nx_graph is None:
            self._nx_graph = nx.from_dict_of_lists(self.graph)
        return self._nx_graph

    @lru_cache(maxsize=1000)
    def _get_reachable_nodes(self, source_node: int) -> frozenset:
        """Cache reachable nodes for each source to avoid recomputation"""
        try:
            reachable = set(
                nx.single_source_shortest_path_length(self.nx_graph, source_node).keys()
            )
            return frozenset(reachable)
        except nx.NetworkXError:
            return frozenset()

    def _get_common_reachable_nodes(self, subquery_nodes: List[int]) -> Set[int]:
        """Get nodes reachable from ALL subquery nodes"""
        if not subquery_nodes:
            return set()

        # Use cached reachability
        reachable_sets = [self._get_reachable_nodes(node) for node in subquery_nodes]

        # Intersection of all reachable sets
        common_reachable = set(reachable_sets[0])
        for reachable_set in reachable_sets[1:]:
            common_reachable &= reachable_set
            if not common_reachable:  # Early exit if no common nodes
                break

        return common_reachable

    def get_possible_placement_nodes_optimized(
        self,
        current_projection: Any,
        primitive_events_per_projection: Dict[str, List[Any]],
        network_data: Dict[int, List[str]],
        index_event_nodes: Dict[str, List[Any]],
        event_distribution_matrix: List[List[Any]],
        placed_subqueries: Dict[Any, int],
    ) -> List[int]:
        """
        Optimized version that combines all filtering steps and uses caching.
        """

        # Step 1: Get base possible nodes (using existing optimized function)
        current_projection_str = str(current_projection)

        possible_nodes = check_possible_placement_nodes_for_input(
            projection=current_projection,
            combination=primitive_events_per_projection[current_projection_str],
            network_data=network_data,
            index_event_nodes=index_event_nodes,
            event_nodes=event_distribution_matrix,
            routing_dict=self.routing_dict,
        )

        # Early exit if no possible nodes or no placement constraints
        if not possible_nodes or not placed_subqueries:
            return self._ensure_deterministic_order(possible_nodes)

        # Step 2: Apply reachability constraints (optimized)
        subquery_nodes = list(placed_subqueries.values())
        common_reachable = self._get_common_reachable_nodes(subquery_nodes)

        # Filter nodes based on reachability
        valid_nodes = [node for node in possible_nodes if node in common_reachable]

        return self._ensure_deterministic_order(valid_nodes)

    def _ensure_deterministic_order(self, nodes: List[int]) -> List[int]:
        """Ensure deterministic ordering of nodes"""
        if not nodes:
            return []

        try:
            return sorted(nodes)
        except Exception:
            return nodes
