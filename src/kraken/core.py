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
from .cost_calculation import get_selection_rate, calculate_costs
from .node_tracker import get_global_event_placement_tracker
from .initialization import initialize_placement_state, setup_run
from .candidate_selection import check_resources, get_all_possible_placement_nodes
from .fallback import get_strategy_recommendation
from .state import PlacementDecision, PlacementDecisionTracker, check_if_projection_has_placed_subqueries, update_tracker

logger = get_kraken_logger(__name__)



def compute_kraken_for_projection(
        self,
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
        single_selectivity: dict,
        projrates: dict,
        graph: networkx.Graph,
        network: list,
        central_eval_plan: list,
        sinks: list[int] = [0]
) -> Any:
    try:
        setup_run()

        has_placed_subqueries = check_if_projection_has_placed_subqueries(
            projection=projection,
            mycombi=mycombi,
            global_tracker=get_global_placement_tracker()
        )

        placement_state = initialize_placement_state(
            combination=combination,
            proj_filter_dict=proj_filter_dict,
            no_filter=no_filter,
            projection=projection,
            graph=graph
        )

        placement_decision_tracker = PlacementDecisionTracker(
            projection=projection
        )

        possible_placement_nodes = get_all_possible_placement_nodes(
            projection=projection,
            placement_state=placement_state,
            network_data=network_data,
            index_event_nodes=index_event_nodes,
            event_nodes=event_nodes
        )

        selection_rate_for_projection = get_selection_rate(
            projection=projection,
            combination_dict=self.h_mycombi,
            selectivities=self.selectivities
        )

        for node in possible_placement_nodes:

            results = calculate_costs(
                placement_node=node,
                projection=projection,
                query_workload=self.query_workload,
                network=network,
                selectivity_rate=selection_rate_for_projection,
                selectivities=self.selectivities,
                combination_dict=self.h_mycombi,
                rates=rates,
                projection_rates=projrates,
                index_event_nodes=index_event_nodes,
                mode=self.config.mode,
                shortest_path_distances=all_pairs,
                sink_nodes=sinks,
                has_placed_subqueries=has_placed_subqueries
            )

            all_push_costs, push_pull_costs, latency, computing_time, transmission_ratio, aquisition_steps = results

            has_enough_resources = check_resources(
                node=node,
                projection=projection,
                network=network,
                combination=combination
            )

            best_strategy = get_strategy_recommendation(
                all_push_costs=all_push_costs,
                push_pull_costs=push_pull_costs,
                has_enough_resources=has_enough_resources
            )

            final_costs = push_pull_costs if best_strategy == 'push_pull' else all_push_costs

            placement_decision = PlacementDecision(
                node=node,
                costs=final_costs,
                strategy=best_strategy,
                all_push_costs=all_push_costs,
                push_pull_costs=push_pull_costs,
                has_sufficient_resources=has_enough_resources,
                plan_details={
                    'computing_time': computing_time,
                    'latency': latency,
                    'transmission_ratio': transmission_ratio,
                    'aquisition_steps': aquisition_steps
                }
            )

            placement_decision_tracker.add_decision(placement_decision)

        best_decision = placement_decision_tracker.get_best_decision()
        update_tracker(
            best_decision=best_decision,
            placement_decision_tracker=placement_decision_tracker,
            projection=projection
        )

        return best_decision

    except Exception as e:
        logger.error(f"Error in compute_kraken_for_projection: {e}")
        raise
