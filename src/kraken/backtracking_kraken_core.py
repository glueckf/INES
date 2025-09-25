from .candidate_selection import get_all_possible_placement_nodes, check_possible_placement_nodes_for_input
from .cost_calculation import calculate_costs
from .determinism import validate_deterministic_inputs
from .global_placement_tracker import get_global_placement_tracker
from .initialization import setup_run, initialize_placement_state
from .latency_calculation import sort_candidate_nodes, get_baseline_latency
from .logging import get_kraken_logger
from .state import check_if_projection_has_placed_subqueries

logger = get_kraken_logger(__name__)


def run_backtracking_kraken_with_latency(
        query_workload,
        pairwise_selectivity,
        dependencies_per_projection,
        simulation_mode,
        processing_order,
        unfolded,
        no_filter,
        filter_by_projection,
        event_distribution_matrix,
        index_event_nodes,
        network_data,
        pairwise_distance_matrix,
        global_event_rates,
        projection_rates_selectivity,
        graph,
        network_data_nodes,
        latency_threshold,
        part_type,
        primitive_events_per_projection,
        routing_dict,
        self,
        sinks=[0],
):
    logger.info("Running Backtracking Kraken with Latency Constraints")

    # Initialize global event placement tracker with network_data
    from .node_tracker import initialize_global_event_tracker

    initialize_global_event_tracker(h_network_data=network_data)

    stack_per_projection = {}  # Initialize stack per projection

    best_placement_stack = []  # Overall placement stack

    for current_projection in processing_order:
        current_projections_depndencies = unfolded[current_projection]

        if part_type:
            # TODO: Implement logic for multinode placement
            pass
        else:
            setup_run()

            global_tracker = get_global_placement_tracker()
            global_tracker.register_parent_processing(current_projection)

            # Check if the current projection has placed subqueries
            has_placed_subqueries = check_if_projection_has_placed_subqueries_stack(
                current_projection=current_projection,
                dependencies_per_projection=dependencies_per_projection,
                stack_per_projection=stack_per_projection
            )

            if has_placed_subqueries:
                logger.info(f"Projection {current_projection} has placed subqueries.")

            # Initialize the stack for backtracking
            stack_per_projection[current_projection] = []

            possible_placement_nodes = get_all_possible_placement_nodes_stack(
                current_projection=current_projection,
                primitive_events_per_projection=primitive_events_per_projection,
                network_data=network_data,
                index_event_nodes=index_event_nodes,
                event_distribution_matrix=event_distribution_matrix,
                routing_dict=routing_dict
            )

            # Apply latency based logic here
            candidate_nodes = sort_candidate_nodes(
                possible_placement_nodes=possible_placement_nodes,
                current_projection=current_projection,
                primitive_events_per_projection=primitive_events_per_projection,
            )

            # Here we check the latency constraint
            baseline_latency = get_baseline_latency(
                current_projection=current_projection,
                stack_per_projection=stack_per_projection,
                dependencies_per_projection=dependencies_per_projection,
            )

            for node in candidate_nodes:

                result = calculate_costs(
                    placement_node=node,
                    current_projection=current_projection,
                    query_workload=query_workload,
                    network_data_nodes=network_data_nodes,
                    pairwise_selectivities=pairwise_selectivity,
                    dependencies_per_projection=dependencies_per_projection,
                    global_event_rates=global_event_rates,
                    projection_rates_selectivity=projection_rates_selectivity,
                    index_event_nodes=index_event_nodes,
                    simulation_mode=simulation_mode,
                    pairwise_distance_matrix=pairwise_distance_matrix,
                    sink_nodes=sinks,
                    has_placed_subqueries=has_placed_subqueries,
                )

                (
                    all_push_costs,
                    push_pull_costs,
                    push_latency,
                    push_pull_latency,
                    computing_time,
                    transmission_ratio,
                    acquisition_steps,
                ) = result

                all_push_latency = push_latency + baseline_latency

                if all_push_costs == push_pull_costs and push_latency == push_pull_latency:
                    # Push pull also uses all push, so we only have all push configuration
                    if all_push_latency > latency_threshold:
                        logger.info(f"Pruning node {node} for projection {current_projection} due to latency")
                        break
                    stack_per_projection[current_projection].append(
                        (
                            node,
                            "all_push",
                            all_push_costs,
                            all_push_latency,
                        )
                    )
                    continue

                # Push pull and all push are different strategies, we have to check both

                if all_push_latency > latency_threshold:
                    # Even with all push, we exceed the latency threshold, so we can prune everything afterwards
                    logger.info(f"Pruning node {node} for projection {current_projection} due to latency")
                    break

                # all push latency is acceptable, so we add it to the stack with other meta information
                stack_per_projection[current_projection].append(
                    (
                        node,
                        "all_push",
                        all_push_costs,
                        all_push_latency,
                    )
                )

                push_pull_latency += baseline_latency

                if push_pull_latency > latency_threshold:
                    # Push-pull latency exceeds the threshold, we skip adding this configuration
                    logger.info(
                        f"Skipping push-pull configuration for node {node} on projection {current_projection} due to latency")
                    continue

                # Now we can add the push_pull configuration as well
                stack_per_projection[current_projection].append(
                    (
                        node,
                        "push_pull",
                        push_pull_costs,
                        push_pull_latency,
                    )
                )

            # Now we are through every possible node for this projection and need to get the best result from the stack
            # Sort the stack by cost first, then by latency, then by the biggest node
            stack_per_projection[current_projection].sort(key=lambda x: (x[2], x[3], -x[0]))

            try:
                best_placement = stack_per_projection[current_projection][0]
            except IndexError:
                # We do not hvae any valid placements in the stack, meaning we cannot go further and need to backtrack
                break

            best_placement_stack[current_projection] = best_placement

            # Now we need to update the tracker

    pass


def check_if_projection_has_placed_subqueries_stack(
        current_projection,
        dependencies_per_projection,
        stack_per_projection
):
    # return if any of the dependencies for this projection are in the stack and have at least one placement
    for dependency in dependencies_per_projection.get(current_projection, []):
        if dependency in stack_per_projection and len(stack_per_projection[dependency]) > 0:
            return True
    return False


def get_all_possible_placement_nodes_stack(
        current_projection,
        primitive_events_per_projection,
        network_data,
        index_event_nodes,
        event_distribution_matrix,
        routing_dict
):
    """
    Get all possible placement nodes for a projection.

    This function finds all nodes where the projection could potentially be placed
    by checking common ancestor requirements and applies deterministic validation.

    Args:
        current_projection: The projection being placed
        stack_per_projection: Current stack of placements per projection
        network_data: The network data structure
        index_event_nodes: List of nodes where events are indexed
        event_distribution_matrix: Matrix indicating event distributions across nodes

    Returns:
        List of possible placement nodes for the current projection
    """

    current_projection_str = str(current_projection)

    possible_placement_nodes = check_possible_placement_nodes_for_input(
        projection=current_projection,
        combination=primitive_events_per_projection[current_projection_str],
        network_data=network_data,
        index_event_nodes=index_event_nodes,
        event_nodes=event_distribution_matrix,
        routing_dict=routing_dict
    )

    possible_placement_nodes = validate_deterministic_inputs(
        possible_placement_nodes, logger
    )

    return possible_placement_nodes
