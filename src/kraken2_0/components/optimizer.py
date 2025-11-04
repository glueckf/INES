"""Placement optimizer for candidate node selection."""

from functools import lru_cache
from typing import Any, Dict, List, Set, Tuple

import networkx as nx


class PlacementOptimizer:
    """
    Component responsible for determining which physical nodes are viable
    candidates for placing a given projection.

    Combines data availability filtering (ancestor-based) with reachability
    filtering from already-placed dependencies.
    """

    def __init__(self, graph: Any, routing_dict: Dict):
        """Initialize optimizer with network topology information.

        Args:
            graph: NetworkX graph representing network topology
            routing_dict: Dictionary containing routing information
        """
        # Handle both dict and NetworkX graph inputs
        if isinstance(graph, dict):
            self.graph = graph
            self._nx_graph = None
        else:
            # Assume it's already a NetworkX graph
            self._nx_graph = graph
            self.graph = None

        self.routing_dict = routing_dict

    @property
    def nx_graph(self):
        """Lazy NetworkX graph creation or return existing graph."""
        if self._nx_graph is None:
            self._nx_graph = nx.from_dict_of_lists(self.graph)
        return self._nx_graph

    def get_possible_placement_nodes_optimized(
        self, projection: Any, placed_subqueries: Dict[Any, int], params: Dict[str, Any]
    ) -> Tuple[List[int], bool, Dict[int, Any]]:
        """Get list of viable placement nodes for a projection.

        Two-phase filtering:
        1. Data availability: Find nodes that are ancestors of required event sources
        2. Reachability: Filter by reachability from placed dependencies
        TODO: I (Ariane) would add 3. Resource Availability: The node has resources that can host the projection

        Args:
            projection: Projection to be placed
            placed_subqueries: Already placed subqueries mapping
            params: Problem parameters dictionary

        Returns:
            List of node IDs that can host this projection or a sub_projection for MNP sorted by ID
        """
        # Phase 1: Get ancestor-based candidates (data availability), missings_by_node (what event type is incomplete at what node)
        possible_nodes = self._get_ancestor_based_candidates(projection, params)

        mnp_possible = bool(possible_nodes[1])
        # Early exit if no possible nodes or no placement constraints
        if not possible_nodes[0]:
            return [], False, {}

        if not placed_subqueries:
            return self._ensure_deterministic_order(possible_nodes[0]), mnp_possible, possible_nodes[1] #sort by ID

        # Phase 2: Apply reachability constraints
        subquery_nodes = list(placed_subqueries.values())
        common_reachable = self._get_common_reachable_nodes(subquery_nodes)

        # Filter nodes based on reachability
        valid_nodes = [node for node in possible_nodes[0] if node in common_reachable]

        # clean missings_by_node (keep only MNP entries that are still valid)
        filtered_missing_by_node = {n: possible_nodes[1][n] for n in valid_nodes if n in possible_nodes[1]}

        # MNP still possible iff at least one valid node remains in that map
        mnp_possible = bool(filtered_missing_by_node)

        return self._ensure_deterministic_order(valid_nodes), mnp_possible, filtered_missing_by_node

    def _get_ancestor_based_candidates(
        self, projection: Any, params: Dict[str, Any]
    ) -> Tuple[List[int], Dict[int, Any]]:
        """Find nodes that are common ancestors to (i) all required event sources for SNP, and (ii) to all required event sources but one which can be incomplete for MNP.

        Args:
            projection: Projection being placed
            params: Problem parameters

        Returns:
             (candidate nodes, is_mnp, missing_by_node)
               - candidate nodes: List of node IDs that can potentially host projection (SNP is subset of MNP)
               - is_mnp: True, iff at least one MNP candidate exist, else False
               - missing_by_node: {node_id -> missing(or better incomplete)_event type} only for MNP

        """
        from src.helper.structures import getNodes

        # Extract parameters
        network_data = params["network_data"]
        index_event_nodes = params["index_event_nodes"]
        event_nodes = params["event_nodes"]
        primitive_events_per_projection = params["primitive_events_per_projection"]

        # Get primitive events for this projection
        projection_str = str(projection)
        combination = primitive_events_per_projection.get(projection_str, [])

        # Non-leaf nodes (fog/cloud nodes that don't produce events)
        non_leaf = [node for node, produced in network_data.items() if not produced]

        # Extract required primitive events
        required_events = set(self._flatten_events(combination))

        # --- Build per-event type source sets
        etb_sources_cache: Dict[Any, List[int]] = {}
        event_sources: Dict[Any, Set[int]] = {}  # {event -> set(nodes that produce it)} # TODO save it because it is a heavy repeating

        for ev in required_events:
            srcs: Set[int] = set()
            for etb in index_event_nodes.get(ev, ()):
                if etb not in etb_sources_cache:
                    etb_sources_cache[etb] = getNodes(etb, event_nodes, index_event_nodes)
                srcs.update(etb_sources_cache[etb])
            event_sources[ev] = srcs

        required_sources: Set[int] = set().union(*event_sources.values()) if event_sources else set()

        # Fast bail-out: if nothing is required, any non-leaf is suitable
        if not required_sources:
            return non_leaf, {}

        # --- Evaluate each destination's coverage
        events_list = list(required_events) # TODO assumption is that this are event types not physical sources like A,B,C
        suitable_mnp_nodes: List[int] = []  # nodes that cover >= num_events
        suitable_snp_nodes: List[int] = []  # nodes that cover all events fully
        missing_by_node: Dict[int, Any] = {}  #

        for dest in non_leaf:
            rd_dest = self.routing_dict.get(dest, {})
            # Build ancestor set for this destination, i.e., from this dest node on to the cloud all nodes should be capable of placement
            ancestor_sources = {
                s for s, info in rd_dest.items()
                if info.get("common_ancestor") == dest
            }

            # Check how many event types are fully covered at 'dest' node
            missing_events = [
                ev for ev in events_list
                if not event_sources[ev].issubset(ancestor_sources)
            ]

            if not missing_events:
                suitable_snp_nodes.append(dest)
                suitable_mnp_nodes.append(dest)
            elif len(missing_events) == 1:
                suitable_mnp_nodes.append(dest)
                missing_by_node[dest] = missing_events[0]

        if suitable_mnp_nodes:
            return suitable_mnp_nodes, missing_by_node

        return suitable_snp_nodes, {}


    def _flatten_events(self, combination: List[Any]) -> List[str]:
        """Extract primitive event names from combination entries.

        Args:
            combination: List of events (trees or primitives)

        Returns:
            List of primitive event names
        """
        out = []
        for ev in combination:
            if hasattr(ev, "children") and hasattr(ev, "leafs"):
                out.extend(ev.leafs())
            else:
                out.append(ev)
        return out

    @lru_cache(maxsize=1000)
    def _get_reachable_nodes(self, source_node: int) -> frozenset:
        """Get all nodes reachable from a source node (cached).

        Args:
            source_node: Starting node ID

        Returns:
            Frozen set of reachable node IDs
        """
        try:
            reachable = set(
                nx.single_source_shortest_path_length(self.nx_graph, source_node).keys()
            )
            return frozenset(reachable)
        except nx.NetworkXError:
            return frozenset()

    def _get_common_reachable_nodes(self, subquery_nodes: List[int]) -> Set[int]:
        """Get nodes reachable from ALL subquery nodes.

        Args:
            subquery_nodes: List of node IDs where subqueries are placed

        Returns:
            Set of node IDs reachable from all subquery nodes
        """
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

    def _ensure_deterministic_order(self, nodes: List[int]) -> List[int]:
        """Ensure deterministic ordering of nodes.

        Args:
            nodes: List of node IDs

        Returns:
            Sorted list of node IDs
        """
        if not nodes:
            return []

        try:
            return sorted(nodes)
        except Exception:
            return nodes
