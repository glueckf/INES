"""
Kraken 2.0 Entry Point and Orchestrator

This module serves as the main entry point for the Kraken 2.0 solver framework.
It orchestrates the problem setup, strategy execution, and result reporting with
a clean, modern interface.
"""

from typing import Any, Dict, List
import time
import uuid

from helper.processCombination_aug import compute_dependencies
from allPairs import create_routing_dict

from src.kraken2_0.problem import PlacementProblem
from src.kraken2_0.state import SolutionCandidate


def run_kraken_solver(
    ines_context: Any,
    strategies_to_run: List[Any],
    latency_threshold: float = None,
) -> Dict[str, Any]:
    """
    Main entry point for the Kraken 2.0 solver framework.

    Orchestrates the entire solving process: setup, execution, and reporting.

    Args:
        ines_context: The INES simulation context object containing all problem data.
        strategies_to_run: List of PlacementAlgorithm enums specifying which strategies to execute.
        latency_threshold: Maximum acceptable latency as a fraction of central latency (or None for no limit).

    Returns:
        Dictionary containing execution results with the following structure:
        {
            "run_id": UUID,
            "strategies": {
                "GREEDY": {"status": "success", "solution": ..., "metrics": {...}},
                ...
            },
            "problem_info": {...},
            "best_solution": SolutionCandidate or None
        }
    """
    run_id = uuid.uuid4()
    start_time = time.time()

    # Phase 1: Setup & Data Gathering
    context = _gather_problem_parameters(ines_context)

    # Calculate processing order using dependency computation
    dependencies = compute_dependencies(
        ines_context, ines_context.h_mycombi, ines_context.h_criticalMSTypes
    )
    processing_order = sorted(dependencies.keys(), key=lambda x: dependencies[x])

    # Phase 2: Problem Instantiation
    problem = PlacementProblem(processing_order, latency_threshold, context)

    # Phase 3: Strategy Execution Loop
    strategy_results = {}

    for strategy_enum in strategies_to_run:
        strategy_name = str(strategy_enum.value) if hasattr(strategy_enum, 'value') else str(strategy_enum)

        # Select and execute strategy
        strategy_start = time.time()
        try:
            strategy = _select_strategy(strategy_enum)
            solution = strategy.solve(problem)
            strategy_end = time.time()

            # Calculate solution metrics
            metrics = _calculate_solution_metrics(solution, problem, ines_context)

            strategy_results[strategy_name] = {
                "status": "success",
                "solution": solution,
                "metrics": metrics,
                "execution_time_seconds": strategy_end - strategy_start,
            }

        except (ValueError, NotImplementedError) as e:
            strategy_end = time.time()
            strategy_results[strategy_name] = {
                "status": "failure",
                "error": str(e),
                "execution_time_seconds": strategy_end - strategy_start,
            }

    # Phase 4: Final Report Assembly
    end_time = time.time()

    # Select best solution (lowest cost among successful strategies)
    best_solution = _select_best_solution(strategy_results)

    final_report = {
        "run_id": str(run_id),
        "total_execution_time_seconds": end_time - start_time,
        "strategies": strategy_results,
        "problem_info": {
            "num_projections": len(processing_order),
            "num_queries": len(ines_context.query_workload),
            "latency_threshold": latency_threshold,
            "network_size": len(ines_context.network),
        },
        "best_solution": best_solution,
    }

    return final_report


def _gather_problem_parameters(ines_context: Any) -> Dict[str, Any]:
    """
    Extract problem parameters from INES context into clean dictionary format.

    Args:
        ines_context: The INES simulation context object.

    Returns:
        Dictionary containing all necessary problem parameters.
    """
    context = {
        # Query and workload
        "query_workload": ines_context.query_workload,
        "dependencies_per_projection": ines_context.h_mycombi,

        # Network topology and routing
        "pairwise_distance_matrix": ines_context.allPairs,
        "graph": ines_context.graph,
        "routing_dict": create_routing_dict(ines_context.graph),

        # Network and event data
        "network_data": ines_context.h_network_data,
        "event_distribution_matrix": ines_context.h_eventNodes,
        "index_event_nodes": ines_context.h_IndexEventNodes,
        "global_event_rates": ines_context.h_rates_data,

        # Projection and selectivity
        "projection_rates_selectivity": ines_context.h_projrates,
        "pairwise_selectivity": ines_context.selectivities,
        "filter_by_projection": ines_context.h_projFilterDict,

        # Node information
        "network_data_nodes": ines_context.network,
        "primitive_events_per_projection": ines_context.h_primitive_events,
        "nodes_per_primitive_event": ines_context.h_nodes,

        # Optimization structures
        "local_rate_lookup": ines_context.h_local_rate_lookup,

        # Latency weighting
        "latency_weighting_factor": getattr(ines_context, "latency_weighting_factor", 1.0),
    }

    return context


def _select_strategy(algorithm_enum: Any):
    """
    Factory function to instantiate the appropriate search strategy.

    Args:
        algorithm_enum: PlacementAlgorithm enum value.

    Returns:
        Instance of the corresponding SearchStrategy implementation.

    Raises:
        NotImplementedError: If the strategy is not yet implemented.
        ValueError: If the algorithm enum is not recognized.
    """

    # TODO: Import actual strategy classes once implemented
    # from src.kraken2_0.search.backtracking import BacktrackingSearch
    # from src.kraken2_0.search.greedy import GreedySearch
    # from src.kraken2_0.search.branch_and_cut import BranchAndCutSearch

    if algorithm_enum:
        return None
    else:
        raise ValueError(
            f"Unknown placement algorithm: {algorithm_enum}. "
            f"Valid options: BACKTRACKING, GREEDY, BRANCH_AND_CUT"
        )


def _calculate_solution_metrics(
    solution: SolutionCandidate,
    problem: PlacementProblem,
    ines_context: Any,
) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a solution.

    Args:
        solution: The solution candidate to analyze.
        problem: The placement problem for context.
        ines_context: INES context for workload information.

    Returns:
        Dictionary containing metrics:
        - total_cost: Total cost across all placements
        - workload_cost: Cost only for workload root queries
        - max_latency: Maximum end-to-end latency
        - num_placements: Number of placement decisions
        - placements_at_cloud: Number of placements at cloud node (0)
        - average_cost_per_placement: Mean cost per placement
    """
    total_cost = solution.cumulative_cost

    # Calculate cost only for workload roots
    workload_set = set(ines_context.query_workload)
    workload_cost = sum(
        info.individual_cost
        for proj, info in solution.placements.items()
        if proj in workload_set
    )

    # Calculate maximum latency
    max_latency = solution.get_critical_path_latency(problem)

    # Count placements at cloud
    placements_at_cloud = sum(
        1 for info in solution.placements.values() if info.node == 0
    )

    # Calculate average cost
    num_placements = len(solution.placements)
    avg_cost = total_cost / num_placements if num_placements > 0 else 0.0

    return {
        "total_cost": total_cost,
        "workload_cost": workload_cost,
        "max_latency": max_latency,
        "num_placements": num_placements,
        "placements_at_cloud": placements_at_cloud,
        "average_cost_per_placement": avg_cost,
    }


def _select_best_solution(strategy_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select the best solution from all successful strategy executions.

    The best solution is defined as the one with the lowest workload cost.

    Args:
        strategy_results: Dictionary of results from each strategy.

    Returns:
        Dictionary containing the best solution with keys:
        - strategy_name: Name of the winning strategy
        - solution: The SolutionCandidate object
        - metrics: The calculated metrics
        Or None if no successful solutions exist.
    """
    successful_results = [
        (name, result)
        for name, result in strategy_results.items()
        if result["status"] == "success"
    ]

    if not successful_results:
        return None

    # Find solution with minimum workload cost
    best_name, best_result = min(
        successful_results,
        key=lambda x: x[1]["metrics"]["workload_cost"]
    )

    return {
        "strategy_name": best_name,
        "solution": best_result["solution"],
        "metrics": best_result["metrics"],
    }
