"""
Example usage of latency-aware KRAKEN placement algorithms.

This module demonstrates how to use the new latency-aware placement
functionality with various constraint scenarios.
"""

import networkx as nx
from typing import List, Dict, Any

from .core import (
    compute_kraken_for_projection_with_latency_constraint,
    compute_kraken_workload_with_latency_constraints,
)
from .state import LatencyConstraint, PlacementContext
from .latency_aware import (
    calculate_all_push_cloud_baseline,
    compute_latency_aware_workload_placement,
)
from .logging import get_kraken_logger

logger = get_kraken_logger(__name__)


def create_example_latency_constraint(
    max_latency_factor: float = 1.5,
) -> LatencyConstraint:
    """
    Create an example latency constraint.

    Args:
        max_latency_factor: Maximum latency factor relative to all-push-to-cloud baseline

    Returns:
        LatencyConstraint: Configured latency constraint
    """
    return LatencyConstraint(
        max_latency_factor=max_latency_factor,
        reference_strategy="all_push_cloud",
        absolute_max_latency=None,  # No absolute limit
    )


def example_single_projection_with_latency_constraint(
    projection: Any,
    query_workload: List[Any],
    network_graph: nx.Graph,
    all_pairs: List[List[float]],
    sinks: List[int] = [0],
    max_latency_factor: float = 1.5,
) -> Any:
    """
    Example: Place a single projection with latency constraint.

    Args:
        projection: The projection to place
        query_workload: Complete query workload
        network_graph: Network topology graph
        all_pairs: Shortest path distances matrix
        sinks: Sink nodes list
        max_latency_factor: Maximum latency factor

    Returns:
        PlacementDecision: Placement result with latency consideration
    """
    logger.info(f"Example: Single projection placement with latency constraint")

    # Create latency constraint
    latency_constraint = create_example_latency_constraint(max_latency_factor)

    # Example parameters (would be provided by calling code)
    selectivities = {}  # Projection selectivities
    h_mycombi = {}  # Projection combinations
    combination = []  # Current projection combination
    proj_filter_dict = {}  # Filter dictionary
    event_nodes = []  # Event emission matrix
    index_event_nodes = {}  # Event to index mapping
    network_data = {}  # Node to events mapping
    mycombi = {}  # All combinations
    rates = {}  # Event rates
    projrates = {}  # Projection rates
    network = []  # Network nodes
    mode = "simulation"  # Simulation mode

    try:
        # Use latency-aware placement
        placement = compute_kraken_for_projection_with_latency_constraint(
            query_workload=query_workload,
            selectivities=selectivities,
            h_mycombi=h_mycombi,
            mode=mode,
            projection=projection,
            combination=combination,
            no_filter=0,
            proj_filter_dict=proj_filter_dict,
            event_nodes=event_nodes,
            index_event_nodes=index_event_nodes,
            network_data=network_data,
            all_pairs=all_pairs,
            mycombi=mycombi,
            rates=rates,
            projrates=projrates,
            graph=network_graph,
            network=network,
            sinks=sinks,
            latency_constraint=latency_constraint,
        )

        if placement:
            logger.info(
                f"Placement result: node {placement.node}, "
                f"strategy {placement.strategy}, "
                f"cost {placement.costs:.2f}, "
                f"latency {placement.latency:.2f}"
            )
        else:
            logger.warning("No valid placement found")

        return placement

    except Exception as e:
        logger.error(f"Error in example placement: {e}")
        return None


def example_workload_placement_with_cascade_optimization(
    query_workload: List[Any],
    projections_in_order: List[Any],
    network_graph: nx.Graph,
    all_pairs: List[List[float]],
    sinks: List[int] = [0],
    max_latency_factor: float = 1.5,
    max_iterations: int = 3,
) -> Dict[Any, Any]:
    """
    Example: Complete workload placement with cascade optimization.

    This demonstrates the full latency-aware KRAKEN algorithm including:
    - Baseline calculation
    - Iterative placement optimization
    - Cascade effect handling

    Args:
        query_workload: Complete query workload
        projections_in_order: Projections in processing order
        network_graph: Network topology graph
        all_pairs: Shortest path distances matrix
        sinks: Sink nodes list
        max_latency_factor: Maximum latency factor
        max_iterations: Maximum optimization iterations

    Returns:
        Dict[Any, PlacementDecision]: Final placement decisions
    """
    logger.info(
        f"Example: Workload placement with cascade optimization "
        f"({len(projections_in_order)} projections, max_latency_factor={max_latency_factor})"
    )

    # Create latency constraint
    latency_constraint = create_example_latency_constraint(max_latency_factor)

    # Example parameters (would be provided by calling code)
    selectivities = {}  # Projection selectivities
    h_mycombi = {}  # Projection combinations
    event_nodes = []  # Event emission matrix
    index_event_nodes = {}  # Event to index mapping
    network_data = {}  # Node to events mapping
    mycombi = {}  # All combinations
    rates = {}  # Event rates
    projrates = {}  # Projection rates
    network = []  # Network nodes
    mode = "simulation"  # Simulation mode

    try:
        # Use workload-level latency-aware placement
        placements = compute_kraken_workload_with_latency_constraints(
            query_workload=query_workload,
            projections_in_order=projections_in_order,
            selectivities=selectivities,
            h_mycombi=h_mycombi,
            mode=mode,
            event_nodes=event_nodes,
            index_event_nodes=index_event_nodes,
            network_data=network_data,
            all_pairs=all_pairs,
            mycombi=mycombi,
            rates=rates,
            projrates=projrates,
            graph=network_graph,
            network=network,
            sinks=sinks,
            latency_constraint=latency_constraint,
            max_iterations=max_iterations,
        )

        # Log results
        logger.info(f"Workload placement completed: {len(placements)} placements")
        for proj, decision in placements.items():
            logger.info(
                f"  {proj}: node {decision.node}, "
                f"strategy {decision.strategy}, "
                f"cost {decision.costs:.2f}, "
                f"latency {decision.latency:.2f}"
            )

        return placements

    except Exception as e:
        logger.error(f"Error in example workload placement: {e}")
        return {}


def example_baseline_calculation(
    projections: List[Any],
    query_workload: List[Any],
    cloud_node: int = 0,
) -> Dict[Any, tuple]:
    """
    Example: Calculate all-push-to-cloud baseline for latency constraints.

    Args:
        projections: List of projections
        query_workload: Complete query workload
        cloud_node: Cloud node identifier

    Returns:
        Dict[Any, tuple]: Mapping of projections to (cost, latency) tuples
    """
    logger.info(f"Example: Baseline calculation for {len(projections)} projections")

    baselines = {}

    # Example parameters (would be provided by calling code)
    combination_dict = {}  # Projection combinations
    rates = {}  # Event rates
    projection_rates = {}  # Projection rates
    selectivities = {}  # Projection selectivities
    index_event_nodes = {}  # Event to index mapping
    shortest_path_distances = []  # Distance matrix
    sink_nodes = [cloud_node]  # Sink nodes
    network = []  # Network nodes
    mode = "simulation"  # Simulation mode

    try:
        for projection in projections:
            baseline_cost, baseline_latency = calculate_all_push_cloud_baseline(
                projection=projection,
                query_workload=query_workload,
                cloud_node=cloud_node,
                combination_dict=combination_dict,
                rates=rates,
                projection_rates=projection_rates,
                selectivities=selectivities,
                index_event_nodes=index_event_nodes,
                shortest_path_distances=shortest_path_distances,
                sink_nodes=sink_nodes,
                network=network,
                mode=mode,
            )

            baselines[projection] = (baseline_cost, baseline_latency)

            logger.info(
                f"Baseline for {projection}: "
                f"cost={baseline_cost:.2f}, latency={baseline_latency:.2f}"
            )

        return baselines

    except Exception as e:
        logger.error(f"Error in baseline calculation: {e}")
        return {}


def example_different_latency_constraints():
    """
    Example: Creating different types of latency constraints.
    """
    logger.info("Example: Different latency constraint configurations")

    # Strict latency constraint (only 20% more latency than baseline)
    strict_constraint = LatencyConstraint(
        max_latency_factor=1.2,
        reference_strategy="all_push_cloud",
    )
    logger.info(
        f"Strict constraint: max_latency_factor={strict_constraint.max_latency_factor}"
    )

    # Relaxed latency constraint (100% more latency allowed)
    relaxed_constraint = LatencyConstraint(
        max_latency_factor=2.0,
        reference_strategy="all_push_cloud",
    )
    logger.info(
        f"Relaxed constraint: max_latency_factor={relaxed_constraint.max_latency_factor}"
    )

    # Constraint with absolute maximum
    absolute_constraint = LatencyConstraint(
        max_latency_factor=1.5,
        reference_strategy="all_push_cloud",
        absolute_max_latency=100.0,  # Hard limit regardless of baseline
    )
    logger.info(
        f"Absolute constraint: max_latency_factor={absolute_constraint.max_latency_factor}, "
        f"absolute_max={absolute_constraint.absolute_max_latency}"
    )

    # Test constraint validation
    baseline_latency = 50.0
    test_latency = 70.0

    for name, constraint in [
        ("strict", strict_constraint),
        ("relaxed", relaxed_constraint),
        ("absolute", absolute_constraint),
    ]:
        acceptable = constraint.is_latency_acceptable(test_latency, baseline_latency)
        logger.info(
            f"{name} constraint: latency {test_latency} with baseline {baseline_latency} "
            f"-> acceptable: {acceptable}"
        )


if __name__ == "__main__":
    # Run examples (would need actual data to execute)
    logger.info(
        "Latency-aware KRAKEN examples - see function implementations for usage patterns"
    )
    example_different_latency_constraints()
