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


def get_baseline_latency(
    current_projection,
    stack_per_projection,
    dependencies_per_projection,
):
    """
    Calculate total latency for a projection and all its subprojections.

    This function traverses the projection tree recursively, calculating latency
    for each subprojection and summing them to get the total latency for the
    current projection's all-push strategy.

    Args:
        latency_threshold: Maximum allowed latency threshold
        result: Placement result containing latency information
        current_projection: The projection to calculate latency for
        stack_per_projection: Stack tracking for backtracking algorithm
        global_tracker: Global placement tracker
        node: Target node for placement
        pairwise_distance_matrix: Matrix of network distances between nodes
        dependencies_per_projection: Mapping of projections to their dependencies
        processing_order: Order in which projections are processed

    Returns:
        float: Total calculated latency, or -1 if exceeds threshold
    """
    dependencies_for_projection = dependencies_per_projection.get(current_projection, [])

    baseline_latency = 0.0

    for dependency in dependencies_for_projection:
        if dependency in stack_per_projection:
            # If any dependency is already in the stack, we take the first element from the stack and it's latency
            dependency_latency = stack_per_projection[dependency][0][3]

            if current_projection.mytype == 'AND':
                baseline_latency = max(baseline_latency, dependency_latency)
            elif current_projection.mytype == 'SEQ':
                baseline_latency += dependency_latency

    return baseline_latency



def _calculate_recursive_subprojection_latency(
    projection, dependencies_per_projection, global_tracker, processing_set, visited
):
    """
    Recursively calculate latency for all nested subprojections.

    Args:
        projection: Current projection to process
        dependencies_per_projection: Mapping of projections to dependencies
        global_tracker: Global placement tracker with decisions
        processing_set: Set of projections being processed
        visited: Set to avoid cycles in dependency graph

    Returns:
        float: Sum of latencies for all nested subprojections
    """
    if projection in visited:
        return 0.0  # Avoid cycles

    visited.add(projection)
    total_nested_latency = 0.0

    try:
        subprojections = dependencies_per_projection.get(projection, [])

        for subproj in subprojections:
            if (
                subproj in processing_set
                and subproj in global_tracker._placement_decisions
            ):
                # Get latency for this subprojection
                subproj_decision = global_tracker._placement_decisions[subproj]
                subproj_latency = 0.0

                if (
                    hasattr(subproj_decision, "plan_details")
                    and "latency" in subproj_decision.plan_details
                ):
                    subproj_latency = subproj_decision.plan_details["latency"]
                elif hasattr(subproj_decision, "latency"):
                    subproj_latency = subproj_decision.latency

                total_nested_latency += subproj_latency

                # Recurse for nested dependencies
                nested_latency = _calculate_recursive_subprojection_latency(
                    subproj,
                    dependencies_per_projection,
                    global_tracker,
                    processing_set,
                    visited.copy(),  # Pass copy to avoid shared state
                )
                total_nested_latency += nested_latency

    finally:
        visited.discard(projection)

    return total_nested_latency
