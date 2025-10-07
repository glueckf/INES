import time
from typing import Any, Dict, List, Tuple, Optional

import networkx as nx

from .candidate_selection import (
    check_possible_placement_nodes_for_input,
)
from .cost_calculation import (
    calculate_costs,
    calculate_push_acquisition_steps,
)
from .determinism import validate_deterministic_inputs
from .event_placement_sorter import EventPlacementSorter
from .event_stack import (
    create_event_stack,
    add_events_to_stack,
)
from .global_placement_tracker import get_global_placement_tracker
from .initialization import setup_run
from .latency_calculation import (
    get_current_config_cost_and_latency,
)
from .logging import setup_backtracking_logger
from .placement_optimizer import PlacementOptimizer

# Initialize logger with file output for detailed debugging
logger = setup_backtracking_logger()


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
    nodes_per_primitive_event,
    self,
    local_rate_lookup: Optional[Dict[str, Dict[int, float]]] = None,
    sinks=[0],
):
    algorithm_start_time = time.time()
    logger.info("=== STARTING BACKTRACKING KRAKEN WITH LATENCY CONSTRAINTS ===")
    logger.info(f"Processing order: {processing_order}")
    logger.info(f"Latency threshold: {latency_threshold}")
    logger.info(f"Total projections to process: {len(processing_order)}")
    logger.info(f"Network has {len(network_data)} nodes")
    logger.info(f"Dependencies per projection: {dependencies_per_projection}")

    # Initialize event stack for tracking events at nodes
    stack_of_events_per_node = create_event_stack()

    stack_per_projection = {}  # Initialize stack per projection
    best_placement_stack = {}  # Overall placement stack - dict mapping projection -> (node, strategy, individual_cost, cumulative_cost, latency, acquisition_steps)

    # Initialize projection index for while loop control
    current_projection_index = 0
    backtrack_count = 0
    total_cost_calculations = 0
    pruned_nodes_count = 0

    placement_optimizer = PlacementOptimizer(graph, routing_dict)
    event_placement_sorter = EventPlacementSorter(event_stack=stack_of_events_per_node)

    while current_projection_index < len(processing_order):
        current_projection = processing_order[current_projection_index]
        stack_per_projection[current_projection] = []
        iteration_start_time = time.time()

        if part_type:
            # TODO: Implement logic for multinode placement
            current_projection_index += 1
            continue
        else:
            setup_run()

            global_tracker = get_global_placement_tracker()
            global_tracker.register_parent_processing(current_projection)

            # Check if the current projection has placed subqueries
            has_placed_subqueries = check_if_projection_has_placed_subqueries_stack(
                current_projection=current_projection,
                dependencies_per_projection=dependencies_per_projection,
                stack_per_projection=stack_per_projection,
            )

            # Initialize the stack for backtracking (only if not already initialized)
            if current_projection not in stack_per_projection:
                stack_per_projection[current_projection] = []

            # Extract placement information from stack
            placed_subqueries = _extract_placed_subqueries_from_stacks(
                current_projection=current_projection,
                best_placement_stack=best_placement_stack,
                dependencies_per_projection=dependencies_per_projection,
            )
            logger.info(f"Extracted placed subqueries: {placed_subqueries}")

            candidate_nodes = get_optimized_candidate_nodes(
                current_projection,
                primitive_events_per_projection,
                network_data,
                index_event_nodes,
                event_distribution_matrix,
                placed_subqueries,
                placement_optimizer,
                event_placement_sorter,
            )

            # Here we check the latency constraint
            current_config_latency, current_config_cost = (
                get_current_config_cost_and_latency(
                    current_projection=current_projection,
                    stack_per_projection=stack_per_projection,
                    dependencies_per_projection=dependencies_per_projection,
                )
            )
            logger.info(
                f"Current configuration - Cost: {current_config_cost}, Latency: {current_config_latency}"
            )

            logger.info(
                f"Starting cost evaluation for {len(candidate_nodes)} candidate nodes"
            )
            evaluation_start_time = time.time()
            nodes_evaluated = 0

            for i, node in enumerate(candidate_nodes):
                node_start_time = time.time()
                logger.debug(f"Evaluating node {node} ({i + 1}/{len(candidate_nodes)})")

                cost_calc_start = time.time()
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
                    nodes_per_primitive_event=nodes_per_primitive_event,
                    has_placed_subqueries=has_placed_subqueries,
                    placed_subqueries=placed_subqueries,
                    local_rate_lookup=local_rate_lookup,
                    stack_of_events_per_node=stack_of_events_per_node,
                )
                cost_calc_time = time.time() - cost_calc_start
                total_cost_calculations += 1
                nodes_evaluated += 1

                (
                    push_cost,
                    push_pull_cost,
                    push_latency,
                    push_pull_latency,
                    computing_time,
                    transmission_ratio,
                    acquisition_steps,
                ) = result

                final_push_latency = push_latency + current_config_latency
                node_eval_time = time.time() - node_start_time

                logger.debug(
                    f"Node {node} evaluation completed in {node_eval_time:.4f}s (cost calc: {cost_calc_time:.4f}s)"
                )
                logger.debug(
                    f"Node {node} costs - Push: {push_cost:.4f}, Push-Pull: {push_pull_cost:.4f}"
                )
                logger.debug(
                    f"Node {node} latencies - Push: {push_latency}, Push-Pull: {push_pull_latency}, Total: {final_push_latency}"
                )

                if push_cost == push_pull_cost and push_latency == push_pull_latency:
                    # Push pull also uses all push, so we only have all push configuration
                    logger.debug(
                        f"Node {node}: Push and push-pull strategies are identical"
                    )
                    if final_push_latency > latency_threshold:
                        logger.info(
                            f"PRUNING: Node {node} for projection {current_projection} due to latency ({final_push_latency} > {latency_threshold})"
                        )
                        pruned_nodes_count += 1
                        break
                    individual_cost = push_cost
                    cumulative_cost = push_cost + current_config_cost
                    stack_per_projection[current_projection].append(
                        (
                            node,
                            "all_push",
                            individual_cost,
                            cumulative_cost,
                            final_push_latency,
                            acquisition_steps,
                        )
                    )
                    logger.debug(
                        f"ADDED to stack: Node {node} with all_push strategy (individual: {individual_cost:.4f}, cumulative: {cumulative_cost:.4f}, latency: {final_push_latency})"
                    )
                    continue
                else:
                    # Push pull and all push are different strategies, we have to check both
                    logger.debug(
                        f"Node {node}: Push and push-pull strategies differ, evaluating both"
                    )

                    if final_push_latency > latency_threshold:
                        # Even with all push, we exceed the latency threshold, so we can prune everything afterwards
                        logger.info(
                            f"PRUNING: Node {node} and all remaining nodes due to latency threshold exceeded ({final_push_latency} > {latency_threshold})"
                        )
                        pruned_nodes_count += len(candidate_nodes) - i
                        break

                    individual_push_cost = push_cost
                    cumulative_push_cost = push_cost + current_config_cost

                    # Calculate all-push acquisition steps
                    all_push_acquisition_steps = calculate_push_acquisition_steps(
                        placement_node=node,
                        projection=current_projection,
                        mycombi=dependencies_per_projection,
                        rates=global_event_rates,
                        projrates=projection_rates_selectivity,
                        placed_subqueries=placed_subqueries,
                        shortest_path_distances=pairwise_distance_matrix,
                        local_rate_lookup=local_rate_lookup,
                    )

                    # all push latency is acceptable, so we add it to the stack with other meta information
                    stack_per_projection[current_projection].append(
                        (
                            node,
                            "all_push",
                            individual_push_cost,
                            cumulative_push_cost,
                            final_push_latency,
                            all_push_acquisition_steps,
                        )
                    )
                    logger.debug(
                        f"ADDED to stack: Node {node} with all_push strategy (individual: {individual_push_cost:.4f}, cumulative: {cumulative_push_cost:.4f}, latency: {final_push_latency})"
                    )

                    final_push_pull_latency = push_pull_latency + current_config_latency

                    if final_push_pull_latency > latency_threshold:
                        # Push-pull latency exceeds the threshold, we skip adding this configuration
                        logger.debug(
                            f"SKIPPING: Node {node} push-pull strategy due to latency ({final_push_pull_latency} > {latency_threshold})"
                        )
                        continue

                    individual_push_pull_cost = push_pull_cost
                    cumulative_push_pull_cost = push_pull_cost + current_config_cost

                    # Now we can add the push_pull configuration as well
                    stack_per_projection[current_projection].append(
                        (
                            node,
                            "push_pull",
                            individual_push_pull_cost,
                            cumulative_push_pull_cost,
                            final_push_pull_latency,
                            acquisition_steps,
                        )
                    )

                evaluation_time = time.time() - evaluation_start_time
                stack_size = len(stack_per_projection[current_projection])
                logger.info(
                    f"Cost evaluation completed: {nodes_evaluated} nodes evaluated, {stack_size} valid candidates found"
                )
                logger.info(
                    f"Evaluation took {evaluation_time:.4f}s (avg {evaluation_time / max(1, nodes_evaluated):.4f}s per node)"
                )
                logger.info(
                    f"Pruned {pruned_nodes_count} nodes due to latency constraints"
                )

                # Sort the stack by cumulative cost first, then by latency, then by the biggest node
                # x[3] is cumulative_cost, x[4] is latency, x[0] is node
                stack_per_projection[current_projection].sort(
                    key=lambda x: (x[3], x[4], -x[0])
                )

                if stack_size > 0:
                    best_candidate = stack_per_projection[current_projection][0]
                    logger.info(
                        f"Best candidate after sorting: Node {best_candidate[0]}, Strategy: {best_candidate[1]}, Individual Cost: {best_candidate[2]:.4f}, Cumulative Cost: {best_candidate[3]:.4f}, Latency: {best_candidate[4]}"
                    )
                    if stack_size > 1:
                        logger.debug(
                            f"Top 5 candidates: {stack_per_projection[current_projection][:5]}"
                        )

            # Try to get the best placement for this projection
            try:
                best_placement = stack_per_projection[current_projection][0]
                best_placement_stack[current_projection] = best_placement

                (
                    best_placement_node,
                    best_placement_strategy,
                    best_placement_individual_cost,
                    best_placement_cumulative_cost,
                    best_placement_latency,
                    best_placement_acquisition_steps,
                ) = best_placement

                # Extract events from acquisition steps
                events = []
                if (
                    best_placement_acquisition_steps
                    and 0 in best_placement_acquisition_steps
                ):
                    events = best_placement_acquisition_steps[0].get(
                        "events_to_pull", []
                    )

                # Add events to the stack
                add_events_to_stack(
                    stack=stack_of_events_per_node,
                    node_id=best_placement_node,
                    events=events,
                    query_id=current_projection,
                    acquisition_type=best_placement_strategy,
                    acquisition_steps=best_placement_acquisition_steps,
                )

                iteration_time = time.time() - iteration_start_time
                logger.info(
                    f"SUCCESS: Projection {current_projection} placed on node {best_placement[0]} with {best_placement[1]} strategy"
                )
                logger.info(
                    f"Placement individual cost: {best_placement[2]:.4f}, cumulative cost: {best_placement[3]:.4f}, latency: {best_placement[4]}, iteration time: {iteration_time:.4f}s"
                )
                current_projection_index += 1
            except IndexError:
                # No valid placements in the stack, need to backtrack
                logger.warning(
                    f"BACKTRACKING: No valid placements found for projection {current_projection}"
                )
                backtrack_count += 1

                try:
                    backtrack_start_time = time.time()
                    backtrack_to_projection = initialize_backtracking(
                        current_projection,
                        stack_per_projection,
                        best_placement_stack,
                        processing_order,
                    )
                    # Set the index to the projection we backtracked to
                    current_projection_index = processing_order.index(
                        backtrack_to_projection
                    )
                    backtrack_time = time.time() - backtrack_start_time
                    logger.info(
                        f"BACKTRACKED to projection {backtrack_to_projection} (index {current_projection_index}) in {backtrack_time:.4f}s"
                    )
                except Exception as e:
                    logger.error(f"FATAL: Backtracking failed with error: {e}")
                    logger.error(
                        f"Algorithm terminating after {backtrack_count} backtracks"
                    )
                    break

    algorithm_time = time.time() - algorithm_start_time

    # Log comprehensive summary
    log_algorithm_summary(
        processing_order=processing_order,
        best_placement_stack=best_placement_stack,
        algorithm_time=algorithm_time,
        backtrack_count=backtrack_count,
        total_cost_calculations=total_cost_calculations,
        pruned_nodes_count=pruned_nodes_count,
    )
    total_cost = 0
    total_cost_all_projections = 0
    max_latency = 0

    # Calculate total costs:
    # 1. Sum individual costs for ALL projections to get true network cost
    # 2. Track which projections are in workload for reporting
    query_workload_set = set(query_workload)
    for projection, placement_info in best_placement_stack.items():
        # Unpack: (node, strategy, individual_cost, cumulative_cost, latency, acquisition_steps)
        node, strategy, individual_cost, cumulative_cost, latency = placement_info[:5]

        # Always sum individual costs to avoid double-counting
        total_cost_all_projections += individual_cost

        if projection in query_workload_set:
            # For workload queries, track their contribution and max latency
            total_cost += individual_cost
            max_latency = max(max_latency, latency)

    return best_placement_stack


def get_optimized_candidate_nodes(
    current_projection,
    primitive_events_per_projection,
    network_data,
    index_event_nodes,
    event_distribution_matrix,
    placed_subqueries,
    placement_optimizer,
    event_placement_sorter,
):
    logger.debug(
        f"Getting optimized candidate nodes for projection {current_projection}"
    )

    possible_nodes_start = time.time()
    possible_placement_nodes = (
        placement_optimizer.get_possible_placement_nodes_optimized(
            current_projection=current_projection,
            primitive_events_per_projection=primitive_events_per_projection,
            network_data=network_data,
            index_event_nodes=index_event_nodes,
            event_distribution_matrix=event_distribution_matrix,
            placed_subqueries=placed_subqueries,
        )
    )
    possible_nodes_time = time.time() - possible_nodes_start
    logger.debug(
        f"Found {len(possible_placement_nodes)} possible placement nodes in {possible_nodes_time:.4f}s"
    )

    sorting_start = time.time()
    candidate_nodes = event_placement_sorter.sort_candidate_nodes_optimized(
        possible_placement_nodes=possible_placement_nodes,
        current_projection=current_projection,
        primitive_events_per_projection=primitive_events_per_projection,
    )
    sorting_time = time.time() - sorting_start
    logger.debug(f"Sorted candidate nodes in {sorting_time:.4f}s")

    return candidate_nodes

    pass


def initialize_backtracking(
    current_projection, stack_per_projection, best_placement_stack, processing_order
):
    logger.info(
        f"BACKTRACK: Starting backtracking from projection {current_projection}"
    )

    current_index = processing_order.index(current_projection)
    if current_index == 0:
        logger.error(
            "BACKTRACK FAILED: Cannot backtrack further, already at the first projection."
        )
        logger.error("No valid solution exists within the given constraints")
        raise Exception(
            "Backtracking failed, no more projections to backtrack to - cannot find valid placement."
        )

    previous_projection = processing_order[current_index - 1]
    logger.info(
        f"BACKTRACK: Moving from projection {current_projection} to {previous_projection}"
    )

    # Remove and update previous projection's placement
    if previous_projection in best_placement_stack:
        old_placement = best_placement_stack[previous_projection]
        del best_placement_stack[previous_projection]
        logger.debug(
            f"BACKTRACK: Removed best placement for {previous_projection}: {old_placement}"
        )

    if previous_projection in stack_per_projection:
        if stack_per_projection[previous_projection]:
            removed_option = stack_per_projection[previous_projection].pop(0)
            logger.debug(
                f"BACKTRACK: Removed option from stack for {previous_projection}: {removed_option}"
            )
            logger.debug(
                f"BACKTRACK: Remaining options for {previous_projection}: {len(stack_per_projection[previous_projection])}"
            )
            if not stack_per_projection[previous_projection]:
                del stack_per_projection[previous_projection]
                logger.debug(f"BACKTRACK: Stack for {previous_projection} is now empty")
        else:
            del stack_per_projection[previous_projection]
            logger.debug(f"BACKTRACK: Deleted empty stack for {previous_projection}")

    # Set new best placement or recurse further
    if (
        previous_projection in stack_per_projection
        and stack_per_projection[previous_projection]
    ):
        new_placement = stack_per_projection[previous_projection][0]
        best_placement_stack[previous_projection] = new_placement
        logger.info(
            f"BACKTRACK: New best placement for {previous_projection}: {new_placement}"
        )
    else:
        logger.warning(
            f"BACKTRACK: No more configurations to try for projection {previous_projection}, continuing backtrack"
        )
        return initialize_backtracking(
            previous_projection,
            stack_per_projection,
            best_placement_stack,
            processing_order,
        )

    # Cleanup subsequent projections
    cleaned_projections = []
    for proj in processing_order[current_index:]:
        if proj in best_placement_stack:
            del best_placement_stack[proj]
            cleaned_projections.append(proj)
        if proj in stack_per_projection:
            del stack_per_projection[proj]

    if cleaned_projections:
        logger.debug(
            f"BACKTRACK: Cleaned up placements for projections: {cleaned_projections}"
        )

    logger.info(f"BACKTRACK: Completed, returning to projection {current_projection}")
    return current_projection


def check_if_projection_has_placed_subqueries_stack(
    current_projection, dependencies_per_projection, stack_per_projection
):
    dependencies = dependencies_per_projection.get(current_projection, [])
    placed_dependencies = [
        dep
        for dep in dependencies
        if dep in stack_per_projection and stack_per_projection[dep]
    ]

    if dependencies:
        logger.debug(
            f"Projection {current_projection} dependencies: {dependencies}, placed: {placed_dependencies}"
        )

    return len(placed_dependencies) > 0


def get_all_possible_placement_nodes_stack(
    current_projection,
    primitive_events_per_projection,
    network_data,
    index_event_nodes,
    event_distribution_matrix,
    routing_dict,
    graph,
    stack_per_projection,
    placed_subqueries,
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
        routing_dict=routing_dict,
    )

    possible_placement_nodes = remove_not_existing_connections(
        possible_placement_nodes=possible_placement_nodes,
        network_data=network_data,
        graph=graph,
        current_projection=current_projection,
        stack_per_projection=stack_per_projection,
        placed_subqueries=placed_subqueries,
    )

    possible_placement_nodes = validate_deterministic_inputs(
        possible_placement_nodes, logger
    )

    return possible_placement_nodes


def remove_not_existing_connections(
    possible_placement_nodes: List[int],
    network_data: Dict[int, List[str]],
    graph: Dict[int, List[int]],
    current_projection: Any = None,
    stack_per_projection: Optional[Dict[Any, List]] = None,
    placed_subqueries: Optional[Dict[Any, int]] = None,
) -> List[int]:
    if current_projection is None or not placed_subqueries:
        return possible_placement_nodes

    reachable_sets = []

    for subquery_node in placed_subqueries.values():
        try:
            reachable = set(
                nx.single_source_shortest_path_length(graph, subquery_node).keys()
            )
            reachable_sets.append(reachable)
        except nx.NetworkXError:
            reachable_sets.append(set())

    if not reachable_sets:
        return possible_placement_nodes

    common_reachable = reachable_sets[0]
    for reachable_set in reachable_sets[1:]:
        common_reachable = common_reachable.intersection(reachable_set)

    valid_nodes = [
        node for node in possible_placement_nodes if node in common_reachable
    ]
    return valid_nodes


def _extract_placed_subqueries_from_stacks(
    current_projection: Any,
    best_placement_stack: Dict[Any, Tuple],
    dependencies_per_projection: Dict[Any, List],
) -> Dict[Any, int]:
    """
    Extract only direct dependencies (depth 1) of current projection from placement stack.

    This optimized version only returns subqueries that are direct dependencies of the
    current projection, avoiding interference from other queries and improving performance.

    Args:
        current_projection: The projection for which we want direct dependencies
        best_placement_stack: Dict mapping projection -> (node, strategy, individual_cost, cumulative_cost, latency, acquisition_steps)
        dependencies_per_projection: Dependencies mapping

    Returns:
        Dict mapping only direct subqueries -> node_id for placed subqueries
    """
    placed_subqueries = {}
    for dependency in dependencies_per_projection.get(current_projection, []):
        placement_tuple = best_placement_stack.get(dependency)
        if placement_tuple:
            # Extract node (index 0) from the tuple
            placed_subqueries[dependency] = placement_tuple[0]

    return placed_subqueries


def log_algorithm_summary(
    processing_order: List[Any],
    best_placement_stack: Dict[Any, Tuple],
    algorithm_time: float,
    backtrack_count: int,
    total_cost_calculations: int,
    pruned_nodes_count: int,
) -> None:
    """
    Log a comprehensive summary of the backtracking algorithm execution.
    """
    logger.info("\\n" + "=" * 80)
    logger.info("BACKTRACKING ALGORITHM EXECUTION SUMMARY")
    logger.info("=" * 80)

    # Basic statistics
    logger.info(f"Total execution time: {algorithm_time:.4f} seconds")
    logger.info(f"Projections to process: {len(processing_order)}")
    logger.info(f"Successfully placed: {len(best_placement_stack)}")
    logger.info(
        f"Success rate: {len(best_placement_stack) / len(processing_order) * 100:.1f}%"
    )

    # Performance metrics
    logger.info("\\nPERFORMANCE METRICS:")
    logger.info(f"  Total backtracks: {backtrack_count}")
    logger.info(f"  Cost calculations: {total_cost_calculations}")
    logger.info(f"  Nodes pruned: {pruned_nodes_count}")
    logger.info(
        f"  Avg time per projection: {algorithm_time / len(processing_order):.4f}s"
    )

    if len(best_placement_stack) == len(processing_order):
        # Solution found - placement structure: (node, strategy, individual_cost, cumulative_cost, latency, acquisition_steps)
        total_individual_cost = sum(
            placement[2] for placement in best_placement_stack.values()
        )
        max_latency = max(placement[4] for placement in best_placement_stack.values())
        avg_cost = total_individual_cost / len(best_placement_stack)

        logger.info("\\nSOLUTION METRICS:")
        logger.info(
            f"  Total individual cost (no double-counting): {total_individual_cost:.6f}"
        )
        logger.info(f"  Average cost per projection: {avg_cost:.6f}")
        logger.info(f"  Maximum latency: {max_latency}")

        # Placement breakdown by strategy
        strategies = {}
        for placement in best_placement_stack.values():
            strategy = placement[1]
            if strategy not in strategies:
                strategies[strategy] = 0
            strategies[strategy] += 1

        logger.info("\\nPLACEMENT STRATEGIES:")
        for strategy, count in strategies.items():
            percentage = count / len(best_placement_stack) * 100
            logger.info(f"  {strategy}: {count} projections ({percentage:.1f}%)")

        # Detailed placement information
        logger.info("\\nDETAILED PLACEMENTS:")
        for proj in processing_order:
            if proj in best_placement_stack:
                placement = best_placement_stack[proj]
                logger.info(
                    f"  Projection {proj}: Node {placement[0]}, Strategy {placement[1]}, Individual Cost {placement[2]:.6f}, Cumulative Cost {placement[3]:.6f}, Latency {placement[4]}"
                )
    else:
        logger.error("\\nINCOMPLETE SOLUTION:")
        logger.error(
            f"  Failed to place {len(processing_order) - len(best_placement_stack)} projections"
        )

        failed_projections = [
            proj for proj in processing_order if proj not in best_placement_stack
        ]
        logger.error(f"  Failed projections: {failed_projections}")

    logger.info("=" * 80)
