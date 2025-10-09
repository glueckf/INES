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
from src.kraken2_0.results_logger import (
    initialize_logging,
    write_detailed_log,
    write_run_results,
)


def run_kraken_solver(
    ines_context: Any,
    strategies_to_run: List[Any],
    enable_detailed_logging: bool = False,
) -> Dict[str, Any]:
    """
    Main entry point for the Kraken 2.0 solver framework.

    Orchestrates the entire solving process: setup, execution, and reporting.

    Args:
        ines_context: The INES simulation context object containing all problem data.
        strategies_to_run: List of PlacementAlgorithm enums specifying which strategies to execute.
        enable_detailed_logging: If True, logs every placement decision to CSV for analysis.

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

    # Initialize detailed logging if enabled
    if enable_detailed_logging:
        initialize_logging()

    # Phase 1: Setup & Data Gathering
    context = _gather_problem_parameters(ines_context)

    # Calculate processing order using dependency computation
    dependencies = compute_dependencies(
        ines_context, ines_context.h_mycombi, ines_context.h_criticalMSTypes
    )

    processing_order = sorted(dependencies.keys(), key=lambda x: dependencies[x])

    # Phase 2: Problem Instantiation
    problem = PlacementProblem(processing_order, context, enable_detailed_logging)

    # Phase 3: Strategy Execution Loop
    strategy_results = {}
    master_log_data = []  # Accumulate logs across all strategies

    for strategy_enum in strategies_to_run:
        strategy_name = (
            str(strategy_enum.value)
            if hasattr(strategy_enum, "value")
            else str(strategy_enum)
        )

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

            # Collect and enrich detailed logs if enabled
            if enable_detailed_logging and hasattr(problem, "detailed_log"):
                enriched_log = _enrich_log_with_solution(
                    problem.detailed_log, solution, str(run_id), strategy_name, problem
                )
                master_log_data.extend(enriched_log)

        except (ValueError, NotImplementedError) as e:
            strategy_end = time.time()
            strategy_results[strategy_name] = {
                "status": "failure",
                "error": str(e),
                "execution_time_seconds": strategy_end - strategy_start,
            }

    # Write detailed logs to Parquet if enabled
    if enable_detailed_logging and master_log_data:
        write_detailed_log(str(run_id), master_log_data)

    # Phase 4: Final Report Assembly
    end_time = time.time()

    # Prepare and write run results summary
    run_results_data = _prepare_run_results_summary(
        str(run_id), strategy_results, problem
    )
    if run_results_data:
        write_run_results(run_results_data)

    # Select best solution (lowest cost among successful strategies)
    best_solution = _select_best_solution(strategy_results)

    final_report = {
        "run_id": str(run_id),
        "total_execution_time_seconds": end_time - start_time,
        "strategies": strategy_results,
        "problem_info": {
            "num_projections": len(processing_order),
            "num_queries": len(ines_context.query_workload),
            "latency_threshold": context.get("latency_threshold", None),
            "network_size": len(ines_context.network),
            "detailed_logging_enabled": enable_detailed_logging,
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
        "all_pairs": ines_context.allPairs,  # Alias for cost calculator
        "graph": ines_context.graph,
        "routing_dict": create_routing_dict(ines_context.graph),
        # Network and event data
        "network_data": ines_context.h_network_data,
        "event_nodes": ines_context.h_eventNodes,
        "event_distribution_matrix": ines_context.h_eventNodes,
        "index_event_nodes": ines_context.h_IndexEventNodes,
        "global_event_rates": ines_context.h_rates_data,
        # Projection and selectivity
        "projection_rates_selectivity": ines_context.h_projrates,
        "pairwise_selectivities": ines_context.selectivities,
        "filter_by_projection": ines_context.h_projFilterDict,
        # Node information
        "network_data_nodes": ines_context.network,
        "primitive_events_per_projection": ines_context.h_primitive_events,
        "nodes_per_primitive_event": ines_context.h_nodes,
        # Sink nodes (cloud is typically node 0)
        "sink_nodes": [0],
        # Optimization structures
        "local_rate_lookup": ines_context.h_local_rate_lookup,
        "sum_of_input_rates_per_query": ines_context.sum_of_input_rates_per_query,
        # Simulation mode
        "simulation_mode": ines_context.config.mode,
        # Latency
        "latency_threshold": ines_context.config.latency_threshold,
        "latency_weighting_factor": ines_context.config.xi,
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
    from src.kraken2_0.search import GreedySearch

    # Get the algorithm name
    algorithm_name = (
        algorithm_enum.value
        if hasattr(algorithm_enum, "value")
        else str(algorithm_enum)
    )

    if algorithm_name == "greedy":
        return GreedySearch()
    elif algorithm_name == "backtracking":
        raise NotImplementedError("Backtracking search not yet implemented")
    elif algorithm_name == "branch_and_cut":
        raise NotImplementedError("Branch and cut search not yet implemented")
    else:
        raise ValueError(
            f"Unknown placement algorithm: {algorithm_enum}. "
            f"Valid options: GREEDY, BACKTRACKING, BRANCH_AND_CUT"
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
        - cumulative_processing_latency: Total processing latency across all placements
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

    # Get cumulative processing latency
    cumulative_processing_latency = solution.cumulative_processing_latency

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
        "cumulative_processing_latency": cumulative_processing_latency,
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
        successful_results, key=lambda x: x[1]["metrics"]["workload_cost"]
    )

    return {
        "strategy_name": best_name,
        "solution": best_result["solution"],
        "metrics": best_result["metrics"],
    }


def _enrich_log_with_solution(
    detailed_log: List[Dict[str, Any]],
    solution: SolutionCandidate,
    run_id: str,
    strategy_name: str,
    problem: Any,
) -> List[Dict[str, Any]]:
    """
    Enrich log entries with solution context and metadata.

    Marks which placement decisions were part of the final solution path
    and adds run identification metadata and problem context.

    Args:
        detailed_log: Raw log entries from problem expansion.
        solution: The final solution candidate.
        run_id: Unique identifier for this run.
        strategy_name: Name of the strategy that generated this solution.
        problem: PlacementProblem instance for accessing workload/processing_order.

    Returns:
        List of enriched log entry dictionaries ready for CSV export.
    """
    # Create a set of (projection, node, strategy) tuples from final solution
    solution_set = {
        (info.projection, info.node, info.strategy)
        for info in solution.placements.values()
    }

    # Format workload and processing_order once for all entries
    workload_str = ";".join(str(q) for q in problem.query_workload)
    processing_order_str = ";".join(str(p) for p in problem.processing_order)

    enriched_log = []
    for entry in detailed_log:
        # Check if this decision was part of the final solution
        decision_tuple = (
            entry["projection"],
            entry["candidate_node"],
            entry["communication_strategy"],
        )
        entry["is_part_of_final_solution"] = decision_tuple in solution_set

        # Add run metadata (same for all entries in this run)
        entry["run_id"] = run_id
        entry["strategy_name"] = strategy_name
        entry["workload"] = workload_str
        entry["processing_order"] = processing_order_str

        enriched_log.append(entry)

    return enriched_log


def _prepare_run_results_summary(
    run_id: str, strategy_results: Dict[str, Any], problem: PlacementProblem
) -> List[Dict[str, Any]]:
    """
    Prepare summary data for run results from strategy executions.

    Extracts key metrics from each strategy execution and formats them
    for storage in the run_results Parquet dataset.

    Args:
        run_id: Unique identifier for this run.
        strategy_results: Dictionary of results from each strategy execution.
        problem: PlacementProblem instance for workload information.

    Returns:
        List of dictionaries containing run result summaries.
    """
    workload_str = ";".join(str(q) for q in problem.query_workload)
    results_data = []

    for strategy_name, result in strategy_results.items():
        result_entry = {
            "run_id": run_id,
            "strategy_name": strategy_name,
            "workload": workload_str,
            "status": result["status"],
            "execution_time_seconds": result["execution_time_seconds"],
        }

        if result["status"] == "success":
            metrics = result["metrics"]
            result_entry.update(
                {
                    "total_cost": metrics["total_cost"],
                    "workload_cost": metrics["workload_cost"],
                    "average_cost_per_placement": metrics["average_cost_per_placement"],
                    "max_latency": metrics["max_latency"],
                    "cumulative_processing_latency": metrics["cumulative_processing_latency"],
                    "num_placements": metrics["num_placements"],
                    "placements_at_cloud": metrics["placements_at_cloud"],
                }
            )
        else:
            # Fill with None for failed strategies
            result_entry.update(
                {
                    "total_cost": None,
                    "workload_cost": None,
                    "average_cost_per_placement": None,
                    "max_latency": None,
                    "cumulative_processing_latency": None,
                    "num_placements": None,
                    "placements_at_cloud": None,
                }
            )

        results_data.append(result_entry)

    return results_data
