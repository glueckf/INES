"""
Candidate node selection and resource checking.

This module handles the selection of candidate nodes for placement
and validation of their resource availability.
"""

from typing import Dict, List, Any
from helper.structures import getNodes
from .logging import get_kraken_logger
from .determinism import validate_deterministic_inputs

logger = get_kraken_logger(__name__)


def check_possible_placement_nodes_for_input(
    projection: Any,
    combination: List[Any],
    network_data: Dict[int, List[str]],
    index_event_nodes: Dict[str, List[Any]],
    event_nodes: List[List[Any]],
    routing_dict: Dict[int, Dict[int, Dict[str, Any]]],
) -> List[int]:
    """
    Faster version:
    - Precompute the primitive events for this combination once.
    - Build the union of all source nodes for all ETBs of those events.
    - For each non-leaf node, check subset: required_sources ⊆ ancestor_sources[node].
    """

    # --- helpers (local to avoid global pollution) ---
    def _flatten_events(comb: List[Any]) -> List[str]:
        """Extract primitive event names from combination entries (trees or primitive)."""
        out = []
        for ev in comb:
            if hasattr(ev, "children") and hasattr(ev, "leafs"):
                out.extend(ev.leafs())
            else:
                out.append(ev)
        return out

    # Cache ETB->sources within this call (getNodes is usually hot)
    etb_sources_cache: Dict[Any, List[int]] = {}

    def _get_etb_sources(etb: Any) -> List[int]:
        srcs = etb_sources_cache.get(etb)
        if srcs is None:
            # getNodes(etb, ...) is assumed to return a list of source node ids
            srcs = getNodes(etb, event_nodes, index_event_nodes)
            etb_sources_cache[etb] = srcs
        return srcs

    # --- start ---
    logger.debug("Analyzing placement options for: %s", projection)
    # non-leaf = fog/cloud nodes (produce no events)
    non_leaf = [node for node, produced in network_data.items() if not produced]

    # 1) Primitive events needed for this projection (dedup once)
    required_events = set(_flatten_events(combination))

    # 2) Union of all source nodes for all ETBs of those events (build once)
    required_sources: set[int] = set()
    for ev in required_events:
        for etb in index_event_nodes.get(ev, ()):
            required_sources.update(_get_etb_sources(etb))

    # Fast bail-out: if nothing is required, any non-leaf is suitable
    if not required_sources:
        logger.info("Result: %d suitable nodes found: %s", len(non_leaf), non_leaf)
        return non_leaf

    # 3) For each candidate destination, precompute its ancestor source set
    #    (i.e., all sources for which destination is the common ancestor)
    #    Then test subset: required_sources ⊆ ancestor_sources[node]
    suitable_nodes: List[int] = []
    for dest in non_leaf:
        rd_dest = routing_dict.get(dest, {})
        # Build ancestor set lazily for this destination
        # NOTE: this is O(degree(dest)) and done once per dest
        ancestor_sources = {
            s for s, info in rd_dest.items() if info.get("common_ancestor") == dest
        }

        # subset check (O(len(required_sources)) set ops in C)
        if required_sources.issubset(ancestor_sources):
            suitable_nodes.append(dest)

    logger.info(
        "Result: %d suitable nodes found: %s", len(suitable_nodes), suitable_nodes
    )
    return suitable_nodes


def check_resources(
    node: int,
    current_projection: Any,
    network_data_nodes: List[Any],
    current_projections_dependencies: List[Any],
) -> bool:
    """
    Function to check if a node has enough resources to place a projection.
    Checks computational power for all-push scenario and memory for push-pull scenario.

    Args:
        node: Node ID to check resources for
        current_projection: Projection object with computing_requirements
        network_data_nodes: List of network nodes with computational_power and memory attributes
        current_projections_dependencies: List of projections in the combination (unused for now)

    Returns:
        bool: True if node has sufficient resources, False otherwise

    Comments:
        As of now, this function checks the given computing_requirements for a projection.
        In the future, they should probably be calculated based on the actual inputs of the projection.
        TODO: Add more complex computing requirements calculation.
    """
    logger.debug(f"Checking resources for node {node}")

    # Get the node from network
    if node >= len(network_data_nodes) or node < 0:
        logger.warning(
            f"Node {node} is out of range (network size: {len(network_data_nodes)})"
        )
        return False

    target_node = network_data_nodes[node]

    try:
        # Check if projection has computing requirements attribute
        if not hasattr(current_projection, "computing_requirements"):
            logger.warning(
                f"Projection {current_projection} has no computing_requirements attribute"
            )
            return False

        # Check computational power for all-push scenario
        if target_node.computational_power < current_projection.computing_requirements:
            logger.debug(
                f"Node {node} insufficient CPU: {target_node.computational_power} < {current_projection.computing_requirements}"
            )
            return False

        # Check memory for push-pull scenario (needs 2x projection requirements)
        if target_node.memory < current_projection.computing_requirements:
            logger.debug(
                f"Node {node} insufficient memory: {target_node.memory} < {2 * current_projection.computing_requirements}"
            )
            return False

        logger.debug(
            f"Node {node} has sufficient resources: CPU={target_node.computational_power}, Memory={target_node.memory}"
        )
        return True

    except Exception as e:
        logger.error(f"Node {node} resource check failed: {e}")
        return False


def get_all_possible_placement_nodes(
    current_projection,
    placement_state,
    network_data,
    index_event_nodes,
    event_distribution_matrix,
) -> List[int]:
    """
    Get all possible placement nodes for a projection.

    This function finds all nodes where the projection could potentially be placed
    by checking common ancestor requirements and applies deterministic validation.

    Args:
        current_projection: The projection being placed
        placement_state: Placement state containing extended_combination and routing_dict
        network_data: Dictionary mapping nodes to event types they produce
        index_event_nodes: Event node index mapping
        event_distribution_matrix: Event-to-node matrix

    Returns:
        List of validated node IDs suitable for placement
    """
    logger.debug(
        f"Finding possible placement nodes for projection {current_projection}"
    )

    # Find possible placement nodes
    possible_placement_nodes = check_possible_placement_nodes_for_input(
        projection=current_projection,
        combination=placement_state["extended_combination"],
        network_data=network_data,
        index_event_nodes=index_event_nodes,
        event_nodes=event_distribution_matrix,
        routing_dict=placement_state["routing_dict"],
    )

    # Validate and sort candidates for deterministic processing
    possible_placement_nodes = validate_deterministic_inputs(
        possible_placement_nodes, logger
    )

    logger.info(
        f"Found {len(possible_placement_nodes)} possible placement nodes: {possible_placement_nodes}"
    )
    return possible_placement_nodes
