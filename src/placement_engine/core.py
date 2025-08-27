"""
Core placement engine facade.

This module provides the main entry point for the placement engine,
maintaining compatibility with the legacy API while providing a clean
internal implementation.
"""

from typing import Any, Dict, List
import networkx
from .state import RuntimeServices
from .determinism import setup_deterministic_environment, log_determinism_info
from .logging import get_placement_logger


def compute_operator_placement_with_prepp(
        self,
        projection: dict,
        combination: list,
        no_filter: int,
        proj_filter_dict: dict,
        event_nodes: list,
        index_event_nodes: dict,
        network_data: dict,
        all_pairs: list,
        mycombi: dict,
        rates: dict,
        single_selectivity: dict,
        projrates: dict,
        graph: networkx.Graph,
        network: list,
        central_eval_plan: list,
        sinks: list[int] = [0]) -> Any:
    """
    Legacy facade — preserves signature. Calls internal engine steps.
    
    This function maintains the exact same signature and behavior as the original
    compute_operator_placement_with_prepp function to ensure complete backward
    compatibility while internally using the modernized placement engine.
    
    Args:
        self: Instance of the class containing all necessary data
        projection: The projection for which the placement is computed
        combination: The combination of event types to consider for the placement
        no_filter: Flag to indicate whether to apply filters or not
        proj_filter_dict: Dictionary containing filters for projections
        event_nodes: Matrix mapping event types to nodes
        index_event_nodes: Indexed dictionary mapping event types to their respective ETBs
        network_data: Dictionary containing data on which node produces which event types
        all_pairs: Matrix containing all pairwise distances between nodes
        mycombi: Dictionary mapping event types to their combinations
        rates: Dictionary containing rates for each event type
        single_selectivity: Dictionary containing selectivity for single event types
        projrates: Dictionary containing rates for each projection
        graph: NetworkX graph representing the network topology
        network: Network object containing all nodes and their respective properties
        central_eval_plan: Central evaluation plan data
        sinks: List of sink nodes to consider for the placement (default: [0] because of single sink placement)
        
    Returns:
        PlacementDecision: Object containing the best placement decision with costs and plan details
    """
    # Initialize determinism harness and logging
    setup_deterministic_environment()
    services = RuntimeServices.create_deterministic()
    logger = get_placement_logger(__name__)

    log_determinism_info(services, logger)
    logger.info(f"Starting placement computation for projection: {projection}")

    # New placement engine implementation
    try:
        from .initialization import initialize_placement_state
        from .candidate_selection import check_possible_placement_nodes_for_input, check_resources
        from .subgraph import extract_subgraph
        from .cost_calculation import calculate_prepp_with_placement, calculate_final_costs_for_sending_to_sinks
        from .fallback import get_strategy_recommendation
        from .determinism import validate_deterministic_inputs
        from .state import PlacementDecision, PlacementDecisionTracker
        from .placement_state import update_placement_state_with_best_decision

        # Initialize placement state
        placement_state = initialize_placement_state(
            combination, proj_filter_dict, no_filter, projection, graph
        )

        # Initialize decision tracker
        placement_decisions = PlacementDecisionTracker(projection)

        # Find possible placement nodes
        possible_placement_nodes = check_possible_placement_nodes_for_input(
            projection,
            placement_state['extended_combination'],
            network_data,
            network,
            index_event_nodes,
            event_nodes,
            placement_state['routing_dict']
        )

        # Validate and sort candidates for deterministic processing
        possible_placement_nodes = validate_deterministic_inputs(possible_placement_nodes, logger)

        # Reverse the order to prioritize nodes closer to the source (legacy behavior)
        possible_placement_nodes.reverse()
        logger.info(f"Evaluating {len(possible_placement_nodes)} placement candidates")

        best_costs = float('inf')
        best_node = None
        best_strategy = 'all_push'

        # Evaluate each candidate node
        for node in possible_placement_nodes:
            logger.info(f"Evaluating node {node}")

            results = calculate_prepp_with_placement(self, node, projection, network)

            if not results:
                logger.warning(f"Node {node} - No results from prepp_with_placement")
                return None

            # If projection is not a subquery, we need to calculate the costs of sending the projection to the cloud
            if projection in self.query_workload and node not in sinks:
                logger.info(f"Node {node} is not a sink, calculating costs for sending projection to cloud")
                # Calculate costs of sending the projection to the cloud
                results = calculate_final_costs_for_sending_to_sinks(
                    cost_results=results,
                    placement_node=node,
                    query_projection=projection,
                    sink_nodes=sinks,
                    projection_rates=self.h_projrates,
                    shortest_path_distances=self.allPairs
                )
            # TODO: Discuss with Ariane since this makes placement @ the cloud more attractive
            #  because push-pull can be used on the whole graph and cuts costs by a lot compared to subgraph placement
            push_pull_costs, computing_time, latency, transmission_ratio, all_push_costs = results

            # Log final calculated costs
            logger.info(f"Final push-pull costs for projection {projection} "
                        f"at node {node}: {push_pull_costs:.2f}")
            logger.info(f"Final all-push costs for projection {projection} "
                        f"at node {node}: {all_push_costs:.2f}")
            logger.info(f"Final latency for projection {projection} "
                        f"at node {node}: {latency:.2f}")
            logger.info(f"Final transmission ratio for projection {projection} "
                        f"at node {node}: {transmission_ratio:.2f}")

            # Check resource availability
            has_enough_resources = check_resources(node, projection, network, combination)

            # # Extract subgraph for this placement node
            # if str(projection) == "AND(A, B, D)":
            #     print("Debug hook")
            # subgraph = extract_subgraph(
            #     node, network, graph, placement_state['extended_combination'],
            #     index_event_nodes, event_nodes, placement_state['routing_dict']
            # )
            #
            # if subgraph:
            #     routing_algo = dict(networkx.all_pairs_shortest_path(subgraph['subgraph']))
            # else:
            #     logger.warning(f"Subgraph extraction failed for node {node}, skipping")
            #     continue
            #
            # # Step 1: Calculate all-push costs (always needed as baseline)
            # central_plan = calculate_all_push_costs_on_subgraph(
            #     self, subgraph, projection
            # )
            #
            # all_push_costs = central_plan[0]
            # central_eval_plan = [central_plan[1], central_plan[3], projection]
            #
            # push_pull_costs = None
            #
            # # Step 2: Calculate push-pull costs if resources allow
            # if has_enough_resources:
            #     try:
            #         push_pull_costs = calculate_prepp_costs_on_subgraph(
            #             self=self,
            #             node=node,
            #             subgraph=subgraph,
            #             projection=projection,
            #             central_eval_plan=central_eval_plan,
            #             all_push_baseline=all_push_costs,
            #             routing_algo=routing_algo
            #         )
            #     except Exception as e:
            #         logger.warning(f"Node {node} - Push-pull calculation failed: {e}")
            #         push_pull_costs = None

            # Determine best strategy for this node
            strategy = get_strategy_recommendation(all_push_costs, push_pull_costs, has_enough_resources)
            final_costs = push_pull_costs if strategy == 'push_pull' else all_push_costs

            logger.info(f"Node {node} - Strategy: {strategy}, Cost: {final_costs:.2f}")

            # Create placement decision
            decision = PlacementDecision(
                node=node,
                costs=final_costs,
                strategy=strategy,
                all_push_costs=all_push_costs,
                push_pull_costs=push_pull_costs,
                has_sufficient_resources=has_enough_resources,
                plan_details={
                    'computing_time': computing_time,
                    'latency': latency,
                    'transmission_ratio': transmission_ratio
                }
            )

            # Track this decision
            placement_decisions.add_decision(decision)

            # Update best placement if this is better
            if final_costs < best_costs:
                best_costs = final_costs
                logger.info(f"New best placement: Node {node} with {strategy} strategy (cost: {final_costs:.2f})")

        # Get the final best decision
        best_decision = placement_decisions.get_best_decision()

        # # Update the structure with the best decision
        # update_placement_state_with_best_decision(
        #     self,
        #     projection,
        #     combination,
        #     no_filter,
        #     proj_filter_dict,
        #     event_nodes,
        #     index_event_nodes,
        #     network_data,
        #     all_pairs,
        #     mycombi,
        #     rates,
        #     single_selectivity,
        #     projrates,
        #     graph,
        #     network,
        #     central_eval_plan,
        #     placement_decisions,
        #     sinks=sinks
        # )



        if best_decision:
            logger.info(f"Final placement decision: {best_decision}")

            # Convert to legacy format for backward compatibility
            # Expected format: (costs, node, longestPath, myProjection, newInstances, Filters)
            # legacy_result = _convert_to_legacy_format(best_decision, placement_decisions)
            # return legacy_result
            return best_decision
        else:
            logger.error("No valid placement found!")
            # Return a fallback decision or raise an exception
            raise RuntimeError("No valid placement nodes found")

    except Exception as e:
        logger.error(f"Error in placement engine: {e}")
        logger.info("Falling back to legacy implementation")

        # Fallback to legacy implementation if new engine fails
        from helper.placement_aug import compute_operator_placement_with_prepp as legacy_impl

        result = legacy_impl(
            self, projection, combination, no_filter, proj_filter_dict, event_nodes,
            index_event_nodes, network_data, all_pairs, mycombi, rates,
            single_selectivity, projrates, graph, network, central_eval_plan
        )

        return result


def _convert_to_legacy_format(best_decision, placement_decisions):
    """
    Convert PlacementDecision to legacy tuple format for backward compatibility.
    
    Legacy format: (costs, node, longestPath, myProjection, newInstances, Filters)
    
    Args:
        best_decision: PlacementDecision object
        placement_decisions: PlacementDecisionTracker object
        
    Returns:
        tuple: Legacy format tuple
    """
    from EvaluationPlan import Projection, Instance

    # Create minimal legacy structures
    costs = best_decision.costs
    node = best_decision.node
    longestPath = 0  # Simplified - could be calculated from subgraph if needed

    # Use the original projection object as the name (from the tracker)
    original_projection = placement_decisions.projection

    # Create a basic projection object with the original projection as name
    myProjection = Projection(
        name=original_projection,  # Use the original projection object, not a string
        combination={},
        sinks=[node],
        spawnedInstances=[],
        Filters=[]
    )

    # Create empty instances list (simplified)
    newInstances = []

    # Create empty filters list (simplified)
    Filters = []

    logger = get_placement_logger(__name__)
    logger.debug(f"Converting to legacy format: costs={costs}, node={node}, projection={original_projection}")

    return (costs, node, longestPath, myProjection, newInstances, Filters)
