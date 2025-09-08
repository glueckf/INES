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
    Check which non-leaf nodes are suitable for placing a projection based on common ancestor requirements.

    This function identifies fog/cloud nodes (non-leaf nodes that don't produce events) where the projection
    could be placed by checking if the node can be a common ancestor for all required ETB sources.
    No resource checks are performed - only topological placement feasibility is considered.

    # TODO: Here we could also consider to do some prefiltering or add some heuristic to reduce the search space.

    Args:
        projection: The projection object for which placement is being computed
        combination: The combination of event types to consider for the placement
        network_data: Dictionary mapping nodes to the event types they produce
        network: List of network node objects
        index_event_nodes: Indexed dictionary mapping event types to their respective ETBs
        event_nodes: Matrix mapping event types to nodes
        routing_dict: Dictionary containing routing information with common ancestors

    Returns:
        list: List of non-leaf node indices where the projection can be placed
    """
    logger.debug(f"Analyzing placement options for: {projection}")
    logger.debug(f"Combination context: {combination}")

    # Get non-leaf nodes (nodes that don't produce events - fog/cloud nodes)
    non_leaf = [node for node, neighbors in network_data.items() if not neighbors]
    logger.debug(
        f"Available fog/cloud nodes: {len(non_leaf)} nodes - {non_leaf[:5]}{'...' if len(non_leaf) > 5 else ''}"
    )

    suitable_nodes = []

    # Check each non-leaf node for placement feasibility
    for destination in non_leaf:
        logger.debug(f"Evaluating node {destination}...")

        skip_destination = False  # Flag to determine if we should skip this destination

        # Check all event types in the combination
        for eventtype in combination:
            if skip_destination:
                break

            logger.debug(f"Processing event type: {eventtype}")

            # Handle Tree objects by getting their leaf events instead of treating them as single entities
            if hasattr(eventtype, "children") and hasattr(eventtype, "leafs"):
                # This is a Tree object (AND, SEQ, etc.), get its leaf events
                leaf_events = eventtype.leafs()
                logger.debug(f"Tree object {eventtype} has leaf events: {leaf_events}")
                for leaf_event in leaf_events:
                    if skip_destination:
                        break
                    logger.debug(f"Processing leaf event: {leaf_event}")
                    # Check all ETBs for this leaf event type
                    for etb in index_event_nodes[leaf_event]:
                        if skip_destination:
                            break

                        logger.debug(f"Analyzing ETB: {etb}")
                        possible_sources = getNodes(etb, event_nodes, index_event_nodes)
                        logger.debug(
                            f"Available sources: {len(possible_sources)} - {possible_sources[:3]}{'...' if len(possible_sources) > 3 else ''}"
                        )

                        # Check each source for this ETB
                        for source in possible_sources:
                            # Use the routing_dict to get the common ancestor
                            common_ancestor = routing_dict[destination][source][
                                "common_ancestor"
                            ]
                            logger.debug(
                                f"Source {source} -> Destination {destination}, Common ancestor: {common_ancestor}"
                            )

                            if common_ancestor != destination:
                                logger.debug(
                                    f"SKIP: Node {destination} cannot be common ancestor for source {source}"
                                )
                                skip_destination = True
                                break

                        if skip_destination:
                            break  # Break out of the etb loop

                    if skip_destination:
                        break  # Break out of the leaf_events loop
            else:
                # This is a primitive event type (string)
                eventtype_key = eventtype
                logger.debug(f"Processing primitive event: {eventtype_key}")

                # Check all ETBs for this event type
                for etb in index_event_nodes[eventtype_key]:
                    if skip_destination:
                        break

                    logger.debug(f"Analyzing ETB: {etb}")
                    possible_sources = getNodes(etb, event_nodes, index_event_nodes)
                    logger.debug(
                        f"Available sources: {len(possible_sources)} - {possible_sources[:3]}{'...' if len(possible_sources) > 3 else ''}"
                    )

                    # Check each source for this ETB
                    for source in possible_sources:
                        # Use the routing_dict to get the common ancestor
                        common_ancestor = routing_dict[destination][source][
                            "common_ancestor"
                        ]
                        logger.debug(
                            f"Source {source} -> Destination {destination}, Common ancestor: {common_ancestor}"
                        )

                        if common_ancestor != destination:
                            logger.debug(
                                f"SKIP: Node {destination} cannot be common ancestor for source {source}"
                            )
                            skip_destination = True
                            break

                    if skip_destination:
                        break  # Break out of the etb loop

            if skip_destination:
                break  # Break out of the eventtype loop

        if not skip_destination:
            suitable_nodes.append(destination)
            logger.debug(f"Node {destination}: SUITABLE - Can serve as common ancestor")
        else:
            logger.debug(f"Node {destination}: SKIPPED - Cannot be common ancestor")

    logger.info(f"Result: {len(suitable_nodes)} suitable nodes found: {suitable_nodes}")
    return suitable_nodes


def check_resources(
    node: int, projection: Any, network: List[Any], combination: List[Any]
) -> bool:
    """
    Function to check if a node has enough resources to place a projection.
    Checks computational power for all-push scenario and memory for push-pull scenario.

    Args:
        node: Node ID to check resources for
        projection: Projection object with computing_requirements
        network: List of network nodes with computational_power and memory attributes
        combination: List of projections in the combination (unused for now)

    Returns:
        bool: True if node has sufficient resources, False otherwise

    Comments:
        As of now, this function checks the given computing_requirements for a projection.
        In the future, they should probably be calculated based on the actual inputs of the projection.
        TODO: Add more complex computing requirements calculation.
    """
    logger.debug(f"Checking resources for node {node}")

    # Get the node from network
    if node >= len(network) or node < 0:
        logger.warning(f"Node {node} is out of range (network size: {len(network)})")
        return False

    target_node = network[node]

    try:
        # Check if projection has computing requirements attribute
        if not hasattr(projection, "computing_requirements"):
            logger.warning(
                f"Projection {projection} has no computing_requirements attribute"
            )
            return False

        # Check computational power for all-push scenario
        if target_node.computational_power < projection.computing_requirements:
            logger.debug(
                f"Node {node} insufficient CPU: {target_node.computational_power} < {projection.computing_requirements}"
            )
            return False

        # Check memory for push-pull scenario (needs 2x projection requirements)
        if target_node.memory < (projection.computing_requirements):
            logger.debug(
                f"Node {node} insufficient memory: {target_node.memory} < {2 * projection.computing_requirements}"
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
    projection, placement_state, network_data, index_event_nodes, event_nodes
) -> List[int]:
    """
    Get all possible placement nodes for a projection.

    This function finds all nodes where the projection could potentially be placed
    by checking common ancestor requirements and applies deterministic validation.

    Args:
        projection: The projection being placed
        placement_state: Placement state containing extended_combination and routing_dict
        network_data: Dictionary mapping nodes to event types they produce
        index_event_nodes: Event node index mapping
        event_nodes: Event-to-node matrix

    Returns:
        List of validated node IDs suitable for placement
    """
    logger.debug(f"Finding possible placement nodes for projection {projection}")

    # Find possible placement nodes
    possible_placement_nodes = check_possible_placement_nodes_for_input(
        projection=projection,
        combination=placement_state["extended_combination"],
        network_data=network_data,
        index_event_nodes=index_event_nodes,
        event_nodes=event_nodes,
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
