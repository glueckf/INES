import uuid

from helper.placement_aug import (
    NEWcomputeCentralCosts,
    ComputeSingleSinkPlacement,
    computeMSplacementCosts,
)
from helper.processCombination_aug import compute_dependencies, getSharedMSinput
import time

from .backtracking_kraken_core import run_backtracking_kraken_with_latency
from .greedy_kraken_core import run_greedy_kraken
from .global_placement_tracker import (
    get_global_placement_tracker,
    reset_global_placement_tracker,
)
from .logging import get_kraken_logger
from .state import get_kraken_timing_tracker, reset_kraken_timing_tracker
from typing import Dict, Any

from allPairs import create_routing_dict

logger = get_kraken_logger(__name__)


def _analyze_placement_metrics(
    placement_decisions: Dict[Any, Any],
    query_workload: list,
    processing_order: list,
    unfolded_projections: Dict[Any, Any],
) -> Dict[str, Any]:
    """
    Analyze placement decisions and calculate comprehensive metrics.

    This function computes various metrics from placement decisions including
    total costs, latency mappings, and maximum workload latency.

    Args:
        placement_decisions: Dictionary mapping projections to their placement decisions
        query_workload: List of root queries in the workload
        processing_order: Ordered list of projections by dependency depth
        unfolded_projections: Dictionary mapping projections to their components

    Returns:
        Dictionary containing:
        - total_costs: Sum of all placement costs
        - latency_mapping: Dictionary mapping projections to latencies
        - workload_roots: List of workload roots with placement decisions
        - max_workload_latency: Maximum latency across all workload roots
        - processing_set: Set of projections in processing order
    """
    # Calculate total costs across all placement decisions
    total_costs = sum(decision.costs for decision in placement_decisions.values())

    # Create latency mapping from placement decisions
    latency_mapping = {
        projection: decision.plan_details.get("latency", 0)
        for projection, decision in placement_decisions.items()
    }

    # Find workload roots that have actual placement decisions
    workload_roots = [
        projection for projection in query_workload if projection in placement_decisions
    ]

    # Create processing set for efficient membership checking
    processing_set = set(processing_order)

    # Calculate maximum latency across workload roots
    max_workload_latency = _calculate_maximum_latency_for_roots(
        roots=workload_roots,
        latency_mapping=latency_mapping,
        unfolded_projections=unfolded_projections,
        processing_set=processing_set,
    )

    return {
        "total_costs": total_costs,
        "max_workload_latency": max_workload_latency,
    }


def _calculate_maximum_latency_for_roots(
    roots: list,
    latency_mapping: Dict[Any, float],
    unfolded_projections: Dict[Any, Any],
    processing_set: set,
) -> float:
    """
    Calculate the maximum latency across all workload root projections.

    This function computes the maximum latency by considering both root projection
    latencies and the latencies of their unfolded subprojections that are being processed.

    Args:
        roots: List of root projections in the workload
        latency_mapping: Dictionary mapping projections to their latencies
        unfolded_projections: Dictionary mapping projections to their components
        processing_set: Set of projections currently being processed

    Returns:
        Maximum latency value across all root projections
    """
    if not roots:
        return 0.0

    def _calculate_root_total_latency(root_projection):
        """Calculate total latency for a single root projection."""
        # Get direct latency for the root projection
        root_latency = latency_mapping.get(root_projection, 0)

        # Get unfolded subprojections for this root
        unfolded_subprojs = unfolded_projections.get(root_projection, ())

        # Find subprojections that are in the processing set
        relevant_subprojs = set(unfolded_subprojs) & processing_set

        # Sum latencies of relevant subprojections
        subproj_latency_sum = sum(
            latency_mapping.get(subproj, 0) for subproj in relevant_subprojs
        )

        return root_latency + subproj_latency_sum

    # Find maximum latency across all root projections
    return max(_calculate_root_total_latency(root) for root in roots)


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


    query_workload = self.query_workload
    filter_by_projection = self.h_projFilterDict
    index_event_nodes = self.h_IndexEventNodes
    pairwise_distance_matrix = self.allPairs
    global_event_rates = self.h_rates_data
    network_data_nodes = self.network
    dependencies_per_projection = self.h_mycombi
    projection_rates_selectivity = self.h_projrates
    event_distribution_matrix = self.h_eventNodes
    graph = self.graph
    pairwise_selectivity = self.selectivities
    simulation_mode = self.config.mode
    event_distribution_matrix = self.h_eventNodes
    index_event_nodes = self.h_IndexEventNodes
    unfolded = self.h_mycombi
    criticalMSTypes = self.h_criticalMSTypes
    network_data = self.h_network_data
    Filters = []
    latency_threshold = self.latency_threshold
    primitive_events_per_projection = self.h_primitive_events

    no_filter = 0  # NO FILTER

    # Access the arguments
    filename = file_path
    number_parents = max_parents

    central_computation_result = NEWcomputeCentralCosts(
        query_workload,
        index_event_nodes,
        pairwise_distance_matrix,
        global_event_rates,
        event_distribution_matrix,
        self.graph,
    )
    (
        central_computation_cost,
        central_computation_node,
        central_computation_longest_path,
        central_computation_routing_dict,
    ) = central_computation_result
    all_push_central_latency = max(pairwise_distance_matrix[central_computation_node])
    number_of_hops = sum(pairwise_distance_matrix[central_computation_node])
    ms_placements = {}
    start_time = time.time()


    # Calculate latency threshold for this run
    if latency_threshold is not None:
        latency_threshold = int(all_push_central_latency * latency_threshold)

    routing_dict = create_routing_dict(graph)

    # Initialize timing tracker for detailed metrics
    reset_kraken_timing_tracker()
    timing_tracker = get_kraken_timing_tracker()
    timing_tracker.start_placement_timing()

    # Initialize global placement tracker for this placement session
    reset_global_placement_tracker()  # Start fresh for each placement calculation
    global_placement_tracker = get_global_placement_tracker()

    # Initialize global event placement tracker with network_data_nodes data
    from .node_tracker import initialize_global_event_tracker

    initialize_global_event_tracker(h_network_data=self.h_network_data)

    hop_latency = {}

    shared_dict = getSharedMSinput(self, unfolded, filter_by_projection)
    dependencies = compute_dependencies(self, unfolded, criticalMSTypes)
    processing_order = sorted(dependencies.keys(), key=lambda x: dependencies[x])
    costs = 0

    integrated_placement_decision_by_projection = {}

    # TODO: This flag currently leaves out MS placement for the integrated approach, as it is not yet implemented
    #  This should be removed and the line in 184 should be commented in once MS placement is implemented
    part_type = False

    if latency_threshold is None:
        for current_projection in (
            processing_order
        ):  # parallelize computation for all projections at the same level
            if set(unfolded[current_projection]) == set(
                current_projection.leafs()
            ):  # initialize hop latency with maximum of children
                hop_latency[current_projection] = 0
            else:
                hop_latency[current_projection] = max(
                    [
                        hop_latency[x]
                        for x in unfolded[current_projection]
                        if x in hop_latency.keys()
                    ]
                )

            # TODO: This should be commented in, once MS placement is implemented for the integrated approach
            # part_type,_,_ = returnPartitioning(self, current_projection, unfolded[current_projection], projection_rates_selectivity ,criticalMSTypes)

            if part_type:
                ms_placements[current_projection] = part_type

                result = computeMSplacementCosts(
                    self,
                    current_projection,
                    unfolded[current_projection],
                    part_type,
                    shared_dict,
                    no_filter,
                    graph,
                )

                additional = result[0]

                costs += additional

                hop_latency[current_projection] += result[1]

                Filters += result[4]

                if current_projection.get_original(
                    query_workload
                ) in query_workload and part_type[0] in list(
                    map(
                        lambda x: str(x),
                        current_projection.get_original(
                            query_workload
                        ).kleene_components(),
                    )
                ):
                    result = ComputeSingleSinkPlacement(
                        current_projection.get_original(query_workload),
                        [current_projection],
                        no_filter,
                    )
                    additional = result[0]
                    costs += additional

            else:
                integrated_optimization_result_for_given_projection = run_greedy_kraken(
                    query_workload=query_workload,
                    pairwise_selectivity=pairwise_selectivity,
                    dependencies_per_projection=dependencies_per_projection,
                    simulation_mode=simulation_mode,
                    current_projection=current_projection,
                    current_projections_dependencies=unfolded[current_projection],
                    no_filter=no_filter,
                    filter_by_projection=filter_by_projection,
                    event_distribution_matrix=event_distribution_matrix,
                    index_event_nodes=index_event_nodes,
                    network_data=network_data,
                    pairwise_distance_matrix=pairwise_distance_matrix,
                    global_event_rates=global_event_rates,
                    projection_rates_selectivity=projection_rates_selectivity,
                    graph=graph,
                    network_data_nodes=network_data_nodes,
                )

                integrated_placement_decision_by_projection[current_projection] = (
                    integrated_optimization_result_for_given_projection
                )
    else:
        # Start latency aware kraken
        logger.info("Starting latency-aware integrated placement...")
        integrated_placement_decision_by_projection = (
            run_backtracking_kraken_with_latency(
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
            )
        )
        pass

    integrated_placement_decision_by_projection = finalize_placement_results(
        self=self,
        placement_decisions_by_projection=integrated_placement_decision_by_projection,
    )

    # Analyze placement decisions and calculate comprehensive metrics
    placement_metrics = _analyze_placement_metrics(
        placement_decisions=integrated_placement_decision_by_projection,
        query_workload=self.query_workload,
        processing_order=processing_order,
        unfolded_projections=unfolded,
    )

    # Extract metrics for backward compatibility and clarity
    costs_for_evaluation_total_workload = placement_metrics["total_costs"]
    max_latency = placement_metrics["max_workload_latency"]

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

    # Get PrePP cache statistics
    from .cost_calculation import _get_prepp_cache_stats

    cache_stats = _get_prepp_cache_stats()

    logger.info(
        f"Timing breakdown - Total: {kraken_execution_time_seconds:.3f}s, "
        f"Placement only: {kraken_execution_time_placement:.3f}s, "
        f"PrePP total: {kraken_execution_time_push_pull:.3f}s"
    )
    logger.info(
        f"PrePP cache stats - Hits: {cache_stats['hits']}, "
        f"Misses: {cache_stats['misses']}, "
        f"Hit rate: {cache_stats['hit_rate_percent']:.1f}%, "
        f"Cache size: {cache_stats['cache_size']}"
    )

    # Calculate algorithm-specific metrics
    placements_at_cloud = len(
        [
            decision
            for decision in integrated_placement_decision_by_projection.values()
            if hasattr(decision, "node") and decision.node == 0
        ]
    )

    # Calculate network_data_nodes topology metrics
    try:
        import networkx as nx

        # Calculate network_data_nodes metrics
        network_clustering_coefficient = nx.average_clustering(graph)
        avg_node_degree = (
            sum(dict(graph.degree()).values()) / len(graph.nodes())
            if len(graph.nodes()) > 0
            else 0
        )

        # Network centralization (based on degree centrality)
        degree_centralities = list(nx.degree_centrality(graph).values())
        if degree_centralities and len(graph.nodes()) > 2:
            max_centrality = max(degree_centralities)
            centrality_sum = sum(max_centrality - c for c in degree_centralities)
            max_possible_sum = (len(graph.nodes()) - 1) * (len(graph.nodes()) - 2)
            network_centralization = (
                centrality_sum / max_possible_sum if max_possible_sum > 0 else 0
            )
        else:
            network_centralization = 0.0

    except Exception as e:
        logger.warning(f"Failed to calculate network_data_nodes topology metrics: {e}")
        network_clustering_coefficient = 0.0
        avg_node_degree = 0.0
        network_centralization = 0.0

    # Calculate query complexity metrics
    try:
        # Query complexity score based on query_workload size, dependency depth, and projections count
        total_projections = len(dependencies)
        dependency_depths = list(dependencies.values())
        max_dependency_length = max(dependency_depths) if dependency_depths else 0

        # Query complexity score (normalized metric combining multiple factors)
        query_complexity_score = (
            len(query_workload) * 0.3  # Workload size influence
            + total_projections * 0.3  # Total projections influence
            + max_dependency_length * 0.4  # Dependency complexity influence
        )

        # Highest query output rate (maximum current_projection rate in processing order)
        highest_query_output_rate = (
            max(
                (projection_rates_selectivity[proj][1] for proj in processing_order),
                default=0.0,
            )
            if projection_rates_selectivity
            else 0.0
        )

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
        "prepp_cache_hits": cache_stats["hits"],
        "prepp_cache_misses": cache_stats["misses"],
        "prepp_cache_hit_rate": cache_stats["hit_rate_percent"],
        "start_time": start_time,
        "end_time": end_time,
        "total_execution_time_seconds": end_time - start_time,
        "push_pull_plan_cost_sum": costs_for_evaluation_total_workload,
        "push_pull_plan_latency": max_latency,
        "central_cost": central_computation_cost,
        "central_hop_latency": all_push_central_latency,
        "number_hops": number_of_hops,
        "workload_size": len(query_workload),
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
        integrated_placement_decision_by_projection, execution_info, query_workload
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
