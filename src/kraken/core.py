"""
Core placement engine facade.

This module provides the main entry point for the placement engine,
maintaining compatibility with the legacy API while providing a clean
internal implementation.
"""

from typing import Any
import networkx
from .logging import get_kraken_logger
from .global_placement_tracker import get_global_placement_tracker
from .cost_calculation import calculate_costs
from .initialization import initialize_placement_state, setup_run
from .candidate_selection import check_resources, get_all_possible_placement_nodes
from .fallback import get_strategy_recommendation
from .state import (
    PlacementDecision,
    PlacementDecisionTracker,
    check_if_projection_has_placed_subqueries,
    update_tracker,
)

logger = get_kraken_logger(__name__)


def compute_kraken_for_projection(
    query_workload:list,
    selectivities: dict,
    h_mycombi: dict,
    mode,
    projection,
    combination: list,
    no_filter: int,
    proj_filter_dict: dict,
    event_nodes: list,
    index_event_nodes: dict,
    network_data: dict,
    all_pairs: list,
    mycombi: dict,
    rates: dict,
    projrates: dict,
    graph: networkx.Graph,
    network: list,
    sinks: list[int] = [0],
) -> Any:
    """Compute optimal joint operator placement and push-pull communication strategy.

    This function implements the core Kraken algorithm for joint optimization of
    operator placement (INEv) and push-pull communication (PrePP) in hierarchical
    fog-cloud distributed complex event processing (DCEP) systems. Unlike sequential
    approaches that optimize placement first and then communication, Kraken considers
    the interdependencies between these strategies to achieve superior performance.

    The algorithm evaluates all feasible placement nodes for a given projection and
    selects the optimal combination of placement location and communication strategy
    (push vs. push-pull) based on a unified cost function that considers network
    transmission costs, latency, and resource constraints.

    Args:
        projection: The current projection to be placed. Projections are processed
            sequentially according to the processing order.
        combination (list): List of subprojections for the current projection.
            Elements can be primitive events (e.g., 'A') or subqueries (e.g., 'AND(A, B)').
        no_filter (int): Binary flag indicating filter presence (0: no filter, 1: has filter).
        proj_filter_dict (dict): Dictionary containing subprojections, their filters,
            and output rates before/after filtering.
        event_nodes (list): Matrix where rows represent Event Type Bundles (ETBs),
            columns represent nodes, and values indicate ETB emission (1 if ETB is
            emitted by node, 0 otherwise). Row indices map to index_event_nodes.
        index_event_nodes (dict): Mapping of primitive events to their ETB indices
            in the event_nodes matrix.
        network_data (dict): Node-to-primitive-events mapping specifying which
            primitive events are emitted by each node (e.g., {0: [], 1: ['A', 'B'], 2: ['C']}).
        all_pairs (list): Precomputed shortest path distances between all node pairs.
            Note: This disregards edge unidirectionality.
        mycombi (dict): Dictionary containing all combinations for each projection
            in processing order.
        rates (dict): Global output rates for each primitive event.
        projrates (dict): Output rates and selectivities as tuples for each projection.
        graph (networkx.Graph): NetworkX graph representation of the network topology.
        network (list): List of all node objects in the network.
        sinks (list[int], optional): List of sink nodes in the network.
            Defaults to [0] (cloud as single sink).

    Returns:
        PlacementDecision: Optimal placement decision for given projection.

    Raises:
        Exception: If any error occurs during placement computation. Errors are logged
            for debugging purposes.

    Note:
        This is the main entry point for the Kraken joint optimization framework.
        The algorithm is designed to be robust and extensible, allowing for future
        integration of more sophisticated placement strategies or decision-making
        algorithms. The function maintains compatibility with the legacy INES API
        while providing improved internal implementation.

    """
    try:
        # Setup for the run, initializes a deterministic environment
        setup_run()

        # Register this parent projection for FIFO conflict resolution
        global_tracker = get_global_placement_tracker()
        global_tracker.register_parent_processing(projection)

        # Check if the current projection has any subqueries that are already placed
        # This influences the placement decisions, since we need to respect already placed subqueries.
        has_placed_subqueries = check_if_projection_has_placed_subqueries(
            projection=projection,
            mycombi=mycombi,
            global_tracker=global_tracker,
        )

        # Placement initialization and state setup.
        placement_state = initialize_placement_state(
            combination=combination,
            proj_filter_dict=proj_filter_dict,
            no_filter=no_filter,
            projection=projection,
            graph=graph,
        )

        # Initialize the placement_decision_tracker for the current projection
        # This tracker will hold all placement decisions evaluated for the projection
        # and help in selecting the best one at the end.
        placement_decision_tracker = PlacementDecisionTracker(projection=projection)

        # Get all possible_placement_nodes for the current projection
        # This returns all nodes where all the projections ETBs are available
        possible_placement_nodes = get_all_possible_placement_nodes(
            projection=projection,
            placement_state=placement_state,
            network_data=network_data,
            index_event_nodes=index_event_nodes,
            event_nodes=event_nodes,
        )

        # Go through all possible placement nodes and calculate the costs
        # For each candidate node, we optimize subprojection placements considering parent context
        for node in possible_placement_nodes:
            # Check if this projection can optimize any of its subprojections for this candidate node
            optimized_subprojections = {}
            if has_placed_subqueries:
                optimized_subprojections = (
                    global_tracker.optimize_subprojection_placement(
                        parent_projection=projection,
                        parent_node=node,
                        shortest_path_distances=all_pairs,
                        projection_rates=projrates,
                    )
                )

            # Calculate the costs for placing the current projection on this node
            # This function respects already placed subqueries if there are any
            # Also respects already locally available events
            # And also adds the final costs if the final query e.g. projection = query in query_workload
            # is placed on a non sink node (not the cloud)
            results = calculate_costs(
                placement_node=node,
                projection=projection,
                query_workload=query_workload,
                network=network,
                selectivities=selectivities,
                combination_dict=h_mycombi,
                rates=rates,
                projection_rates=projrates,
                index_event_nodes=index_event_nodes,
                mode=mode,
                shortest_path_distances=all_pairs,
                sink_nodes=sinks,
                has_placed_subqueries=has_placed_subqueries,
            )

            # Initialize the relevant variables from the results we got back
            (
                all_push_costs,
                push_pull_costs,
                latency,
                computing_time,
                transmission_ratio,
                acquisition_steps,
            ) = results

            # Check if the current node has enough resources to place the projection there.
            # This check is done after calculating the costs, since prepp might potentially lower
            # the ressource requirements making nodes that would otherwise be infeasible feasible
            has_enough_resources = check_resources(
                node=node,
                projection=projection,
                network=network,
                combination=combination,
            )

            # For this node and this projection, given the costs and the ressource availability
            # we get a recommendation which strategy we should use
            best_strategy = get_strategy_recommendation(
                all_push_costs=all_push_costs,
                push_pull_costs=push_pull_costs,
                has_enough_resources=has_enough_resources,
            )

            # Update the costs according to the best strategy
            base_costs = (
                push_pull_costs if best_strategy == "push_pull" else all_push_costs
            )

            # Calculate heuristic adjustment for subprojection transfer costs
            # This is ONLY used for selection, not stored as actual costs
            heuristic_adjustment = 0.0
            if optimized_subprojections:
                for subquery, optimization in optimized_subprojections.items():
                    # Add transfer cost as heuristic penalty
                    transfer_cost = optimization.get("transfer_cost", 0.0)
                    heuristic_adjustment += transfer_cost

                    # If placement changed, we might get better performance
                    if optimization.get("placement_changed", False):
                        logger.debug(
                            f"Heuristic: subprojection {subquery} transfer cost: {transfer_cost:.2f}"
                        )

            # Final costs for comparison include heuristic adjustment
            final_costs_for_comparison = base_costs + heuristic_adjustment

            # Track this placement decision for the current node
            placement_decision = PlacementDecision(
                node=node,
                costs=final_costs_for_comparison,  # Include heuristic for selection
                strategy=best_strategy,
                all_push_costs=all_push_costs,
                push_pull_costs=push_pull_costs,
                has_sufficient_resources=has_enough_resources,
                plan_details={
                    "computing_time": computing_time,
                    "latency": latency,
                    "transmission_ratio": transmission_ratio,
                    "acquisition_steps": acquisition_steps,
                    "base_costs": base_costs,  # Store original costs
                    "heuristic_adjustment": heuristic_adjustment,
                },
            )

            # Add optimized subprojections information to the decision
            if optimized_subprojections:
                placement_decision.optimized_subprojections = optimized_subprojections

            placement_decision_tracker.add_decision(placement_decision)

        # Once we checked every possible placement node for the current projection
        # we get the best decision from the placement_decision_tracker
        best_decision = placement_decision_tracker.get_best_decision()

        # Then we need to update our trackers to reflect this placement decision for the next run.
        # This includes updating where this query is placed, but also updating the events this query brings to the node.

        # Update the best decision to use base costs (remove heuristic adjustment)
        if best_decision and "base_costs" in best_decision.plan_details:
            base_costs = best_decision.plan_details["base_costs"]
            heuristic_adjustment = best_decision.plan_details.get(
                "heuristic_adjustment", 0.0
            )

            # Reset costs to base costs (no heuristic)
            best_decision.costs = base_costs

            logger.debug(
                f"Selected placement for {projection}: Node {best_decision.node} "
                f"(base cost: {base_costs:.2f}, heuristic adjustment: {heuristic_adjustment:.2f})"
            )

        # Apply any subprojection placement changes if this parent can optimize them
        if has_placed_subqueries and hasattr(best_decision, "optimized_subprojections"):
            for (
                subquery,
                optimization,
            ) in best_decision.optimized_subprojections.items():
                if not optimization.get("locked", True) and optimization.get(
                    "placement_changed", False
                ):
                    # This subprojection placement actually changed
                    # Update the global tracker with the new placement decision
                    old_tracker = global_tracker.get_placement_decisions(subquery)
                    if old_tracker and "optimized_decision" in optimization:
                        # Update the best decision with the new placement decision
                        old_tracker.best_decision = optimization["optimized_decision"]

                    # Lock this subprojection now that we've optimized it
                    global_tracker.lock_subprojection_placement(subquery)

                    logger.info(
                        f"Re-optimized subprojection {subquery} for parent {projection}: "
                        f"moved to node {optimization['placement_node']} "
                        f"(placement cost: {optimization['optimized_decision'].costs:.2f})"
                    )
                elif not optimization.get("locked", True):
                    # Subprojection can be optimized but placement didn't change
                    # Just lock it to prevent future changes
                    global_tracker.lock_subprojection_placement(subquery)
                    logger.debug(
                        f"Locked subprojection {subquery} for parent {projection} "
                        f"(no placement change needed)"
                    )

        update_tracker(
            best_decision=best_decision,
            placement_decision_tracker=placement_decision_tracker,
            projection=projection,
        )

        # Finally we return the best decision for this projection
        return best_decision

    except Exception as e:
        logger.error(f"Error in compute_kraken_for_projection: {e}")
        raise
