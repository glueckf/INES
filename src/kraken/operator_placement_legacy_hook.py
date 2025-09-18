import uuid

from helper.placement_aug import (
    NEWcomputeCentralCosts,
    ComputeSingleSinkPlacement,
    computeMSplacementCosts,
)
from helper.processCombination_aug import compute_dependencies, getSharedMSinput
import time
from .core import compute_kraken_for_projection
from .global_placement_tracker import (
    get_global_placement_tracker,
    reset_global_placement_tracker,
)
from .logging import get_kraken_logger
from .state import get_kraken_timing_tracker, reset_kraken_timing_tracker
import csv
import os
from typing import Dict, Any

logger = get_kraken_logger(__name__)


def format_results_for_comparison(
    results_dict: Dict, execution_info: Dict, workload: list
) -> Dict[str, Any]:
    """
    Format placement results in a clean, machine-readable format for comparison.

    Args:
        results_dict: Dictionary containing placement results for each projection
        execution_info: Dictionary containing execution metadata

    Returns:
        Dictionary with structured results for easy comparison
    """
    formatted_results = {
        "metadata": execution_info,
        "placements": {},
        "summary": {
            "total_projections": len(results_dict),
            "total_cost": 0,
            "successful_placements": 0,
            "failed_placements": 0,
        },
    }

    for projection, result in results_dict.items():
        projection_str = str(projection)

        if hasattr(result, "costs") and hasattr(result, "node"):
            # New placement engine result format
            formatted_results["placements"][projection_str] = {
                "placement_node": result.node,
                "total_cost": result.costs,
                "strategy": getattr(result, "strategy", "unknown"),
                "all_push_cost": getattr(result, "all_push_costs", None),
                "push_pull_cost": getattr(result, "push_pull_costs", None),
                "has_sufficient_resources": getattr(
                    result, "has_sufficient_resources", None
                ),
                "plan_details": getattr(result, "plan_details", {}),
                "success": True,
            }
            if projection in workload:
                formatted_results["summary"]["total_cost"] += result.costs
            formatted_results["summary"]["successful_placements"] += 1
        else:
            # Handle other result formats or errors
            formatted_results["placements"][projection_str] = {
                "success": False,
                "result_type": str(type(result)),
                "result_data": str(result) if result else None,
            }
            formatted_results["summary"]["failed_placements"] += 1

    return formatted_results


def calculate_integrated_approach(self, file_path: str, max_parents: int):
    workload = self.query_workload
    projFilterDict = self.h_projFilterDict
    IndexEventNodes = self.h_IndexEventNodes
    allPairs = self.allPairs
    rates = self.h_rates_data
    network = self.network
    mycombi = self.h_mycombi
    projrates = self.h_projrates
    EventNodes = self.h_eventNodes
    G = self.graph
    selectivities = self.selectivities
    mode = self.config.mode
    EventNodes = self.h_eventNodes
    IndexEventNodes = self.h_IndexEventNodes
    unfolded = self.h_mycombi
    criticalMSTypes = self.h_criticalMSTypes
    network_data = self.h_network_data
    Filters = []

    noFilter = 0  # NO FILTER

    # Access the arguments
    filename = file_path
    number_parents = max_parents

    central_computation_result = NEWcomputeCentralCosts(
        workload, IndexEventNodes, allPairs, rates, EventNodes, self.graph
    )
    (
        central_computation_cost,
        central_computation_node,
        central_computation_longest_path,
        central_computation_routing_dict,
    ) = central_computation_result
    centralHopLatency = max(allPairs[central_computation_node])
    numberHops = sum(allPairs[central_computation_node])
    MSPlacements = {}
    start_time = time.time()

    # Initialize timing tracker for detailed metrics
    reset_kraken_timing_tracker()
    timing_tracker = get_kraken_timing_tracker()
    timing_tracker.start_placement_timing()

    # Initialize global placement tracker for this placement session
    reset_global_placement_tracker()  # Start fresh for each placement calculation
    global_placement_tracker = get_global_placement_tracker()

    # Initialize global event placement tracker with network data
    from .node_tracker import initialize_global_event_tracker

    initialize_global_event_tracker(h_network_data=self.h_network_data)

    hopLatency = {}

    sharedDict = getSharedMSinput(self, unfolded, projFilterDict)
    dependencies = compute_dependencies(self, unfolded, criticalMSTypes)
    processingOrder = sorted(dependencies.keys(), key=lambda x: dependencies[x])
    costs = 0

    integrated_placement_decision_by_projection = {}

    """ 
    NOTE from FINN GLÜCK 10.09.2025: 
    For some reason in some edge cases projections with really high output rates reach the top of the processing order. 
    We do not want them in our workload, at least for single node queries, that's why we filter them out here. 
    The heuristic for the filtering is: 
    
    Sum of input rates < projection output rate 
    
    Example: 
    
    SEQ(A, B) with 'AB' = 0.5, R(A) = 1000, R(B) = 5
    
    sum of input rates = R(A) + R(B) = 1005
    
    projection output rate = R(A) * R(B) * 'AB' = 1000 * 5 * 0.5 = 2500
    
    1005 < 2500 -> filter out
    """

    """ 
    NOTE FROM Finn Glück 12.09.2025: 
    Actually no subprojections with to high output rates reach the processing order. 
    Those high rate projections are always part of the workload and can therefore not be filtered out here. 
    However the problem for INES remains, 
    as these projections still get placed inside the network and not the cloud where they should be placed.
    """
    # result = update_processing_order_with_heuristics(
    #     query_workload=workload,
    #     combinations=mycombi,
    #     processing_order=processingOrder,
    #     proj_filter_dict=projFilterDict,
    #     rates=rates,
    #     projection_rates=projrates,
    # )
    #
    # (processingOrder, projrates, projFilterDict, mycombi) = result

    # TODO: This flag currently leaves out MS placement for the integrated approach, as it is not yet implemented
    #  This should be removed and the line in 184 should be commented in once MS placement is implemented
    partType = False

    for projection in (
        processingOrder
    ):  # parallelize computation for all projections at the same level
        if set(unfolded[projection]) == set(
            projection.leafs()
        ):  # initialize hop latency with maximum of children
            hopLatency[projection] = 0
        else:
            hopLatency[projection] = max(
                [hopLatency[x] for x in unfolded[projection] if x in hopLatency.keys()]
            )

        # TODO: This should be commented in, once MS placement is implemented for the integrated approach
        # partType,_,_ = returnPartitioning(self, projection, unfolded[projection], projrates ,criticalMSTypes)

        if partType:
            MSPlacements[projection] = partType

            result = computeMSplacementCosts(
                self,
                projection,
                unfolded[projection],
                partType,
                sharedDict,
                noFilter,
                G,
            )

            additional = result[0]

            costs += additional

            hopLatency[projection] += result[1]

            Filters += result[4]

            if projection.get_original(workload) in workload and partType[0] in list(
                map(
                    lambda x: str(x),
                    projection.get_original(workload).kleene_components(),
                )
            ):
                result = ComputeSingleSinkPlacement(
                    projection.get_original(workload), [projection], noFilter
                )
                additional = result[0]
                costs += additional

        else:
            integrated_optimization_result_for_given_projection = (
                compute_kraken_for_projection(
                    workload,
                    selectivities,
                    mycombi,
                    mode,
                    projection,
                    unfolded[projection],
                    noFilter,
                    projFilterDict,
                    EventNodes,
                    IndexEventNodes,
                    network_data,
                    allPairs,
                    mycombi,
                    rates,
                    projrates,
                    G,
                    network,
                )
            )

            integrated_placement_decision_by_projection[projection] = (
                integrated_optimization_result_for_given_projection
            )

    integrated_placement_decision_by_projection = finalize_placement_results(
        self=self,
        placement_decisions_by_projection=integrated_placement_decision_by_projection,
    )

    decisions = integrated_placement_decision_by_projection

    # 1) Total costs: single C-level pass
    costs_for_evaluation_total_workload = sum(d.costs for d in decisions.values())

    # 2) Precompute latency per projection once
    latency_of = {p: d.plan_details.get("latency", 0) for p, d in decisions.items()}

    # 3) Consider only workload roots that actually have decisions
    roots = [p for p in self.query_workload if p in decisions]

    # 4) Set for O(1) membership when filtering unfolded subprojections
    processing_set = set(processingOrder)

    # 5) Max latency over roots: root latency + latencies of unfolded children in processingOrder
    #    Use set(...) on unfolded[root] to dedupe like your original code
    get_unfolded = unfolded.get
    max_latency = (
        max(
            (
                latency_of.get(root, 0)
                + sum(
                    latency_of.get(sub, 0)
                    for sub in (set(get_unfolded(root, ())) & processing_set)
                )
            )
            for root in roots
        )
        if roots
        else 0
    )

    kraken_simulation_id = uuid.uuid4()

    end_time = time.time()
    totaltime = str(end_time - start_time)[:6]

    # Calculate detailed timing metrics
    (
        kraken_execution_time_seconds,
        kraken_execution_time_placement,
        kraken_execution_time_push_pull,
        prepp_call_count,
        placement_evaluations_count,
    ) = timing_tracker.finalize_placement_timing()

    logger.info(
        f"Timing breakdown - Total: {kraken_execution_time_seconds:.3f}s, "
        f"Placement only: {kraken_execution_time_placement:.3f}s, "
        f"PrePP total: {kraken_execution_time_push_pull:.3f}s"
    )

    # Calculate algorithm-specific metrics
    placements_at_cloud = len([
        decision for decision in integrated_placement_decision_by_projection.values()
        if hasattr(decision, "node") and decision.node == 0
    ])

    # Calculate network topology metrics
    try:
        import networkx as nx
        # Calculate network metrics
        network_clustering_coefficient = nx.average_clustering(G)
        avg_node_degree = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0

        # Network centralization (based on degree centrality)
        degree_centralities = list(nx.degree_centrality(G).values())
        if degree_centralities and len(G.nodes()) > 2:
            max_centrality = max(degree_centralities)
            centrality_sum = sum(max_centrality - c for c in degree_centralities)
            max_possible_sum = (len(G.nodes()) - 1) * (len(G.nodes()) - 2)
            network_centralization = centrality_sum / max_possible_sum if max_possible_sum > 0 else 0
        else:
            network_centralization = 0.0

    except Exception as e:
        logger.warning(f"Failed to calculate network topology metrics: {e}")
        network_clustering_coefficient = 0.0
        avg_node_degree = 0.0
        network_centralization = 0.0

    # Calculate query complexity metrics
    try:
        # Query complexity score based on workload size, dependency depth, and projections count
        total_projections = len(dependencies)
        dependency_depths = list(dependencies.values())
        max_dependency_length = max(dependency_depths) if dependency_depths else 0

        # Query complexity score (normalized metric combining multiple factors)
        query_complexity_score = (
            len(workload) * 0.3 +  # Workload size influence
            total_projections * 0.3 +  # Total projections influence
            max_dependency_length * 0.4  # Dependency complexity influence
        )

        # Highest query output rate (maximum projection rate in processing order)
        highest_query_output_rate = max((projrates[proj][1] for proj in processingOrder), default=0.0) if projrates else 0.0

    except Exception as e:
        logger.warning(f"Failed to calculate query complexity metrics: {e}")
        query_complexity_score = 0.0
        highest_query_output_rate = 0.0
        max_dependency_length = 0

    # Prepare execution metadata
    execution_info = {
        "experiment_id": kraken_simulation_id,
        "file_path": filename,
        "max_parents": number_parents,
        "execution_time_seconds": float(totaltime),
        "kraken_execution_time_seconds": kraken_execution_time_seconds,
        "kraken_execution_time_placement": kraken_execution_time_placement,
        "kraken_execution_time_push_pull": kraken_execution_time_push_pull,
        "prepp_call_count": prepp_call_count,
        "placement_evaluations_count": placement_evaluations_count,
        "start_time": start_time,
        "end_time": end_time,
        "total_execution_time_seconds": end_time - start_time,
        "push_pull_plan_cost_sum": costs_for_evaluation_total_workload,
        "push_pull_plan_latency": max_latency,
        "central_cost": central_computation_cost,
        "central_hop_latency": centralHopLatency,
        "number_hops": numberHops,
        "workload_size": len(workload),
        "placements_at_cloud": placements_at_cloud,
        "network_clustering_coefficient": network_clustering_coefficient,
        "network_centralization": network_centralization,
        "avg_node_degree": avg_node_degree,
        "query_complexity_score": query_complexity_score,
        "highest_query_output_rate": highest_query_output_rate,
        "projection_dependency_length": max_dependency_length,
        "global_tracker_entries": len(global_placement_tracker._placement_history)
        if global_placement_tracker
        else 0,
    }

    # Format results for comparison
    formatted_results = format_results_for_comparison(
        integrated_placement_decision_by_projection, execution_info, workload
    )

    result = {
        "kraken_simulation_id": kraken_simulation_id,
        "integrated_placement_decision_by_projection": integrated_placement_decision_by_projection,
        "formatted_results": formatted_results,
    }

    return result


def finalize_placement_results(self, placement_decisions_by_projection):
    """
    Aggregate placement costs by recursively following acquisition steps and adding subprojection costs.
    """
    from collections import defaultdict
    import copy

    # 1) Map "string name" -> projection object once
    proj_by_str = {str(p): p for p in placement_decisions_by_projection}

    # 2) Build adjacency: projection -> list of subprojections (keep multiplicity)
    deps = defaultdict(list)
    for proj, decision in placement_decisions_by_projection.items():
        steps = getattr(decision, "plan_details", {}).get("aquisition_steps", {}) or {}
        for step in steps.values():
            for ev in step.get("events_to_pull", ()):
                sub = proj_by_str.get(ev)
                if sub is not None:
                    deps[proj].append(sub)

    # 3) DFS with memoization (O(N+E)); handle cycles like original (add 0 on cycle)
    memo = {}
    visiting = set()
    get_dec = placement_decisions_by_projection.get

    def total_cost(proj):
        if proj in memo:
            return memo[proj]
        if (
            proj in visiting
        ):  # cycle guard → count nothing, like your visited-check + 0.0
            return 0.0
        visiting.add(proj)

        dec = get_dec(proj)
        base = dec.costs if dec is not None else 0.0
        acc = base
        for child in deps.get(proj, ()):
            acc += total_cost(child)

        visiting.remove(proj)
        memo[proj] = acc
        return acc

    # 4) Build finalized dict; copy only when cost changes
    finalized = {}
    for proj, dec in placement_decisions_by_projection.items():
        tc = total_cost(proj)
        if tc == dec.costs:
            finalized[proj] = dec
        else:
            nd = copy.copy(dec)  # shallow copy preserves type/attrs
            nd.original_costs = dec.costs
            nd.aggregated_additional_cost = tc - dec.costs
            nd.costs = tc
            finalized[proj] = nd

    return finalized
def calculate_placements_at_cloud(integrated_operator_placement_results):
    """
    Calculate the number of projections placed at the cloud node (node 0).

    Args:
        integrated_operator_placement_results: Complete results dict containing all placement data

    Returns:
        int: Count of projections placed at the cloud node
    """
    # Loop through each projection's placement decision and increment count if placed at cloud (node 0)
    placed_at_cloud = 0
    placement_decisions = integrated_operator_placement_results.get(
        "integrated_placement_decision_by_projection", {}
    )
    for projection, decision in placement_decisions.items():
        if hasattr(decision, "node") and decision.node == 0:
            placed_at_cloud += 1

    return placed_at_cloud
