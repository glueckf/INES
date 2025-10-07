from .logging import get_kraken_logger
from .node_tracker import get_global_event_placement_tracker

logger = get_kraken_logger(__name__)


def sort_candidate_nodes(
    possible_placement_nodes,
    current_projection,
    primitive_events_per_projection,
):
    logger.info("Sorting candidate nodes based on latency constraints")

    try:
        # We need to sort to possible placement nodes based on their available events for this projection.
        global_event_placement_tracker = get_global_event_placement_tracker()

        possible_placement_nodes.reverse()  # Reverse to prioritize last added nodes

        placement_nodes = []

        # Convert projection to string key if needed
        projection_key = str(current_projection)

        if projection_key not in primitive_events_per_projection:
            return possible_placement_nodes

        current_projection_primitive_events = primitive_events_per_projection[
            projection_key
        ]

        for node in possible_placement_nodes:
            events_at_node = global_event_placement_tracker.get_events_at_node(node)
            if events_at_node == current_projection_primitive_events:
                node = possible_placement_nodes.pop(node)
                placement_nodes.append(node)

        placement_nodes += possible_placement_nodes

        return placement_nodes
    except Exception as e:
        logger.error(f"Error in sort_candidate_nodes: {e}")
        raise e


def get_current_config_cost_and_latency(
    current_projection,
    stack_per_projection,
    dependencies_per_projection,
):
    """
    Calculate total latency and cost for a projection's dependencies.

    This function examines the direct dependencies of a projection and extracts
    their latencies and cumulative costs from the placement stack.

    Stack entry structure: (node, strategy, individual_cost, cumulative_cost, latency, acquisition_steps)

    Args:
        current_projection: The projection to calculate latency for
        stack_per_projection: Stack tracking for backtracking algorithm
        dependencies_per_projection: Mapping of projections to their dependencies

    Returns:
        tuple: (current_config_latency, current_config_cost)
            - current_config_latency: Max latency among dependencies
            - current_config_cost: Sum of cumulative costs from dependencies
    """
    dependencies_for_projection = dependencies_per_projection.get(
        current_projection, []
    )

    current_config_latency = 0.0
    current_config_cost = 0.0

    for dependency in dependencies_for_projection:
        if dependency in stack_per_projection:
            # Stack structure: (node, strategy, individual_cost, cumulative_cost, latency, acquisition_steps)
            # Index 4 is latency, index 3 is cumulative_cost
            dependency_latency = stack_per_projection[dependency][0][4]
            dependency_cumulative_cost = stack_per_projection[dependency][0][3]

            current_config_latency = max(current_config_latency, dependency_latency)
            current_config_cost += dependency_cumulative_cost

    return current_config_latency, current_config_cost
