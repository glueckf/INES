"""
Latency-aware KRAKEN placement algorithms.

This module implements the latency-aware extensions to the KRAKEN placement
algorithm, including baseline calculation, cascade effect handling, and
iterative optimization with latency constraints.
"""

from typing import Any, Dict, List, Tuple, Optional
import networkx as nx

from .logging import get_kraken_logger
from .cost_calculation import calculate_costs
from .candidate_selection import get_all_possible_placement_nodes
from .fallback import get_strategy_recommendation
from .state import (
    LatencyConstraint,
    PlacementContext,
    PlacementDecision,
    PlacementDecisionTracker,
    EventAvailabilityUpdate,
)
from .node_tracker import get_global_event_placement_tracker

logger = get_kraken_logger(__name__)


def calculate_all_push_cloud_baseline(
    projection: Any,
    query_workload: List[Any],
    cloud_node: int,
    combination_dict: Dict[Any, Any],
    rates: Dict[str, float],
    projection_rates: Dict[Any, Tuple],
    selectivities: Dict[Any, float],
    index_event_nodes: Dict[str, List[str]],
    shortest_path_distances: List[List[float]],
    sink_nodes: List[int],
    network: List[Any],
    mode: Any,
) -> Tuple[float, float]:
    """
    Calculate the all-push-to-cloud baseline cost and latency.

    This establishes the reference point for latency constraints by computing
    the cost and latency of placing all projections at the cloud node.

    Args:
        projection: The projection to evaluate
        query_workload: Complete query workload
        cloud_node: The cloud node identifier
        combination_dict: Dictionary containing all combinations
        rates: Global output rates for each primitive event
        projection_rates: Output rates and selectivities for each projection
        selectivities: Global selectivities for each projection
        index_event_nodes: Mapping of primitive events to ETB indices
        shortest_path_distances: Precomputed shortest path distances
        sink_nodes: List of sink nodes
        network: List of all node objects
        mode: Simulation mode

    Returns:
        Tuple[float, float]: (baseline_cost, baseline_latency)
    """
    logger.debug(f"Computing all-push-to-cloud baseline for {projection}")

    try:
        results = calculate_costs(
            placement_node=cloud_node,
            projection=projection,
            query_workload=query_workload,
            network=network,
            selectivities=selectivities,
            combination_dict=combination_dict,
            rates=rates,
            projection_rates=projection_rates,
            index_event_nodes=index_event_nodes,
            mode=mode,
            shortest_path_distances=shortest_path_distances,
            sink_nodes=sink_nodes,
            has_placed_subqueries=False,  # Baseline assumes no prior placements
        )

        all_push_costs, push_pull_costs, latency, _, _, _ = results
        baseline_cost = all_push_costs  # Use all-push for baseline
        baseline_latency = latency

        logger.debug(
            f"All-push-to-cloud baseline: cost={baseline_cost:.2f}, "
            f"latency={baseline_latency:.2f}"
        )

        return baseline_cost, baseline_latency

    except Exception as e:
        logger.error(f"Failed to compute all-push-to-cloud baseline: {e}")
        # Return high values as fallback
        return float("inf"), float("inf")


def filter_candidates_by_latency_constraint(
    candidates: List[int],
    projection: Any,
    latency_constraint: LatencyConstraint,
    baseline_latency: float,
    context: PlacementContext,
    query_workload: List[Any],
    network: List[Any],
    selectivities: Dict[Any, float],
    combination_dict: Dict[Any, Any],
    rates: Dict[str, float],
    projection_rates: Dict[Any, Tuple],
    index_event_nodes: Dict[str, List[str]],
    mode: Any,
    has_placed_subqueries: bool = False,
) -> List[Tuple[int, float]]:
    """
    Filter placement candidates by latency constraint.

    Args:
        candidates: List of candidate nodes
        projection: The projection being placed
        latency_constraint: The latency constraint to enforce
        baseline_latency: The baseline latency for comparison
        context: Placement context with network information
        query_workload: Complete query workload
        network: List of all node objects
        selectivities: Global selectivities
        combination_dict: Dictionary containing all combinations
        rates: Global output rates
        projection_rates: Output rates and selectivities
        index_event_nodes: Mapping of primitive events to ETB indices
        mode: Simulation mode
        has_placed_subqueries: Whether subqueries have been placed

    Returns:
        List[Tuple[int, float]]: List of (node_id, latency) pairs that meet constraints
    """
    logger.debug(
        f"Filtering {len(candidates)} candidates by latency constraint "
        f"(baseline={baseline_latency:.2f}, max_factor={latency_constraint.max_latency_factor})"
    )

    valid_candidates = []

    for node in candidates:
        try:
            results = calculate_costs(
                placement_node=node,
                projection=projection,
                query_workload=query_workload,
                network=network,
                selectivities=selectivities,
                combination_dict=combination_dict,
                rates=rates,
                projection_rates=projection_rates,
                index_event_nodes=index_event_nodes,
                mode=mode,
                shortest_path_distances=context.shortest_path_distances,
                sink_nodes=context.sink_nodes,
                has_placed_subqueries=has_placed_subqueries,
            )

            _, _, latency, _, _, _ = results

            if latency_constraint.is_latency_acceptable(latency, baseline_latency):
                valid_candidates.append((node, latency))
                logger.debug(f"Node {node}: latency={latency:.2f} (acceptable)")
            else:
                logger.debug(
                    f"Node {node}: latency={latency:.2f} "
                    f"(exceeds constraint: {baseline_latency * latency_constraint.max_latency_factor:.2f})"
                )

        except Exception as e:
            logger.warning(f"Failed to evaluate latency for node {node}: {e}")
            continue

    logger.info(
        f"Latency filtering: {len(valid_candidates)}/{len(candidates)} candidates meet constraint"
    )
    return valid_candidates


def detect_cascade_effects(
    new_placement: PlacementDecision,
    current_placements: Dict[Any, PlacementDecision],
    projection: Any,
    network_graph: nx.Graph,
) -> List[EventAvailabilityUpdate]:
    """
    Detect cascade effects from a new placement decision.

    When a projection is placed, it makes its output events available at that node,
    potentially affecting the cost calculations for subsequent projections.

    Args:
        new_placement: The new placement decision
        current_placements: Currently placed projections
        projection: The projection being placed
        network_graph: Network topology graph

    Returns:
        List[EventAvailabilityUpdate]: Updates to event availability
    """
    logger.debug(
        f"Detecting cascade effects for {projection} placed at node {new_placement.node}"
    )

    cascade_effects = []

    # Get events produced by this projection
    try:
        from .cost_calculation import get_events_for_projection

        new_events = list(get_events_for_projection(projection))

        if new_events:
            update = EventAvailabilityUpdate(
                node_id=new_placement.node,
                events=new_events,
                projection=projection,
                strategy=new_placement.strategy,
            )
            cascade_effects.append(update)

            logger.debug(
                f"Cascade effect: {len(new_events)} events now available at node {new_placement.node}"
            )

    except Exception as e:
        logger.warning(f"Could not determine events for projection {projection}: {e}")

    return cascade_effects


def find_projections_affected_by_cascade(
    event_update: EventAvailabilityUpdate,
    remaining_projections: List[Any],
    combination_dict: Dict[Any, Any],
) -> List[Any]:
    """
    Find projections that are affected by cascade event availability changes.

    Args:
        event_update: The event availability update
        remaining_projections: Projections not yet placed
        combination_dict: Dictionary containing projection combinations

    Returns:
        List[Any]: Projections affected by the cascade
    """
    affected_projections = []

    for proj in remaining_projections:
        try:
            # Check if this projection uses any of the newly available events
            if proj in combination_dict:
                combination = combination_dict[proj]
                # Simple check: if any event name appears in the combination string
                combination_str = str(combination)
                for event in event_update.events:
                    if event in combination_str:
                        affected_projections.append(proj)
                        break
        except Exception as e:
            logger.debug(f"Could not check cascade effect for projection {proj}: {e}")

    logger.debug(f"Cascade affects {len(affected_projections)} remaining projections")
    return affected_projections


def get_strategy_recommendation_with_latency(
    all_push_costs: float,
    push_pull_costs: Optional[float],
    all_push_latency: float,
    push_pull_latency: Optional[float],
    max_allowed_latency: float,
    has_enough_resources: bool,
    latency_weight: float = 0.3,
) -> str:
    """
    Get strategy recommendation considering latency constraints.

    Args:
        all_push_costs: Cost of all-push strategy
        push_pull_costs: Cost of push-pull strategy
        all_push_latency: Latency of all-push strategy
        push_pull_latency: Latency of push-pull strategy
        max_allowed_latency: Maximum allowed latency
        has_enough_resources: Whether node has sufficient resources
        latency_weight: Weight for latency in multi-objective optimization

    Returns:
        str: Recommended strategy ('all_push' or 'push_pull')
    """
    # First filter by latency constraint
    valid_strategies = []

    if all_push_latency <= max_allowed_latency:
        valid_strategies.append(("all_push", all_push_costs, all_push_latency))

    if (
        push_pull_costs is not None
        and push_pull_latency is not None
        and push_pull_latency <= max_allowed_latency
        and has_enough_resources
    ):
        valid_strategies.append(("push_pull", push_pull_costs, push_pull_latency))

    # If no strategies meet latency constraint, select closest to constraint
    if not valid_strategies:
        logger.warning("No strategies meet latency constraint, selecting closest")

        all_push_violation = all_push_latency - max_allowed_latency
        strategies_by_violation = [("all_push", all_push_violation)]

        if push_pull_latency is not None and has_enough_resources:
            push_pull_violation = push_pull_latency - max_allowed_latency
            strategies_by_violation.append(("push_pull", push_pull_violation))

        # Return strategy with smallest latency violation
        return min(strategies_by_violation, key=lambda x: x[1])[0]

    # Among valid strategies, use multi-objective selection
    if len(valid_strategies) == 1:
        return valid_strategies[0][0]

    # Normalize costs and latencies for multi-objective comparison
    max_cost = max(s[1] for s in valid_strategies)
    min_cost = min(s[1] for s in valid_strategies)
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0

    scores = []
    for strategy, cost, latency in valid_strategies:
        normalized_cost = (cost - min_cost) / cost_range
        normalized_latency = latency / max_allowed_latency

        # Weighted combination: lower is better
        score = (
            1 - latency_weight
        ) * normalized_cost + latency_weight * normalized_latency
        scores.append((score, strategy))

    best_strategy = min(scores, key=lambda x: x[0])[1]

    logger.debug(
        f"Multi-objective strategy selection: chose {best_strategy} "
        f"from {len(valid_strategies)} valid options"
    )

    return best_strategy


def compute_latency_aware_placement_for_projection(
    projection: Any,
    query_workload: List[Any],
    context: PlacementContext,
    selectivities: Dict[Any, float],
    combination_dict: Dict[Any, Any],
    rates: Dict[str, float],
    projection_rates: Dict[Any, Tuple],
    index_event_nodes: Dict[str, List[str]],
    network_data: Dict[int, List[str]],
    event_nodes: List[List[int]],
    network: List[Any],
    mode: Any,
    has_placed_subqueries: bool = False,
    placement_state: Any = None,
) -> Optional[PlacementDecision]:
    """
    Compute latency-aware placement for a single projection.

    Args:
        projection: The projection to place
        query_workload: Complete query workload
        context: Placement context with latency constraints
        selectivities: Global selectivities
        combination_dict: Dictionary containing all combinations
        rates: Global output rates
        projection_rates: Output rates and selectivities
        index_event_nodes: Mapping of primitive events to ETB indices
        network_data: Node-to-primitive-events mapping
        event_nodes: ETB emission matrix
        network: List of all node objects
        mode: Simulation mode
        has_placed_subqueries: Whether subqueries have been placed
        placement_state: Current placement state

    Returns:
        Optional[PlacementDecision]: Best placement decision or None if no valid placement
    """
    logger.debug(f"Computing latency-aware placement for {projection}")

    # Get all possible placement nodes
    possible_placement_nodes = get_all_possible_placement_nodes(
        projection=projection,
        placement_state=placement_state,
        network_data=network_data,
        index_event_nodes=index_event_nodes,
        event_nodes=event_nodes,
    )

    if not possible_placement_nodes:
        logger.warning(f"No possible placement nodes found for {projection}")
        return None

    placement_decision_tracker = PlacementDecisionTracker(projection=projection)

    # If latency constraint is specified, filter candidates
    if context.latency_constraint and context.baseline_latency:
        valid_candidates = filter_candidates_by_latency_constraint(
            candidates=possible_placement_nodes,
            projection=projection,
            latency_constraint=context.latency_constraint,
            baseline_latency=context.baseline_latency,
            context=context,
            query_workload=query_workload,
            network=network,
            selectivities=selectivities,
            combination_dict=combination_dict,
            rates=rates,
            projection_rates=projection_rates,
            index_event_nodes=index_event_nodes,
            mode=mode,
            has_placed_subqueries=has_placed_subqueries,
        )

        if not valid_candidates:
            logger.warning(f"No candidates meet latency constraint for {projection}")
            return None

        candidate_nodes = [node for node, _ in valid_candidates]
    else:
        candidate_nodes = possible_placement_nodes

    # Evaluate each candidate node
    for node in candidate_nodes:
        try:
            results = calculate_costs(
                placement_node=node,
                projection=projection,
                query_workload=query_workload,
                network=network,
                selectivities=selectivities,
                combination_dict=combination_dict,
                rates=rates,
                projection_rates=projection_rates,
                index_event_nodes=index_event_nodes,
                mode=mode,
                shortest_path_distances=context.shortest_path_distances,
                sink_nodes=context.sink_nodes,
                has_placed_subqueries=has_placed_subqueries,
            )

            (
                all_push_costs,
                push_pull_costs,
                latency,
                computing_time,
                transmission_ratio,
                acquisition_steps,
            ) = results

            # Check resource availability
            from .candidate_selection import check_resources

            has_enough_resources = check_resources(
                node=node,
                projection=projection,
                network=network,
                combination=[],  # TODO: Pass actual combination if available
            )

            # Get strategy recommendation with latency awareness
            if context.latency_constraint and context.baseline_latency:
                max_allowed_latency = (
                    context.baseline_latency
                    * context.latency_constraint.max_latency_factor
                )

                # Special case: latency factor of 1.0 should force all-push strategy
                if context.latency_constraint.max_latency_factor <= 1.0:
                    logger.debug(
                        f"Latency factor {context.latency_constraint.max_latency_factor} <= 1.0, forcing all-push strategy"
                    )
                    best_strategy = "all_push"
                else:
                    # Calculate per-strategy latencies (simplified approach)
                    all_push_latency = latency
                    push_pull_latency = latency if push_pull_costs is not None else None

                    best_strategy = get_strategy_recommendation_with_latency(
                        all_push_costs=all_push_costs,
                        push_pull_costs=push_pull_costs,
                        all_push_latency=all_push_latency,
                        push_pull_latency=push_pull_latency,
                        max_allowed_latency=max_allowed_latency,
                        has_enough_resources=has_enough_resources,
                    )
            else:
                # Fall back to standard strategy selection
                best_strategy = get_strategy_recommendation(
                    all_push_costs=all_push_costs,
                    push_pull_costs=push_pull_costs,
                    has_enough_resources=has_enough_resources,
                )

            # Select costs based on strategy
            final_costs = (
                push_pull_costs if best_strategy == "push_pull" else all_push_costs
            )

            # Create placement decision with latency information
            placement_decision = PlacementDecision(
                node=node,
                costs=final_costs,
                strategy=best_strategy,
                all_push_costs=all_push_costs,
                push_pull_costs=push_pull_costs,
                has_sufficient_resources=has_enough_resources,
                latency=latency,
                all_push_latency=latency,
                push_pull_latency=latency if push_pull_costs is not None else None,
                plan_details={
                    "computing_time": computing_time,
                    "latency": latency,
                    "transmission_ratio": transmission_ratio,
                    "acquisition_steps": acquisition_steps,
                },
            )

            placement_decision_tracker.add_decision(placement_decision)

        except Exception as e:
            logger.error(f"Error evaluating node {node} for {projection}: {e}")
            continue

    return placement_decision_tracker.get_best_decision()


def compute_latency_aware_workload_placement(
    query_workload: List[Any],
    projections_in_order: List[Any],
    context: PlacementContext,
    selectivities: Dict[Any, float],
    combination_dict: Dict[Any, Any],
    rates: Dict[str, float],
    projection_rates: Dict[Any, Tuple],
    index_event_nodes: Dict[str, List[str]],
    network_data: Dict[int, List[str]],
    event_nodes: List[List[int]],
    network: List[Any],
    mode: Any,
    max_iterations: int = 3,
) -> Dict[Any, PlacementDecision]:
    """
    Compute latency-aware placement for entire workload with cascade handling.

    This is the main algorithm that implements iterative placement optimization
    to handle cascade effects where one placement decision affects others.

    Args:
        query_workload: Complete query workload
        projections_in_order: List of projections in processing order
        context: Placement context with latency constraints
        selectivities: Global selectivities
        combination_dict: Dictionary containing all combinations
        rates: Global output rates
        projection_rates: Output rates and selectivities
        index_event_nodes: Mapping of primitive events to ETB indices
        network_data: Node-to-primitive-events mapping
        event_nodes: ETB emission matrix
        network: List of all node objects
        mode: Simulation mode
        max_iterations: Maximum number of optimization iterations

    Returns:
        Dict[Any, PlacementDecision]: Final placement decisions for all projections
    """
    logger.info(
        f"Starting latency-aware workload placement for {len(projections_in_order)} projections "
        f"(max_iterations={max_iterations})"
    )

    # Step 1: Calculate baseline if latency constraint is specified
    if context.latency_constraint and context.baseline_latency is None:
        logger.info("Computing all-push-to-cloud baseline for latency constraint")
        total_baseline_cost = 0.0
        total_baseline_latency = 0.0

        for proj in projections_in_order:
            baseline_cost, baseline_latency = calculate_all_push_cloud_baseline(
                projection=proj,
                query_workload=query_workload,
                cloud_node=context.cloud_node,
                combination_dict=combination_dict,
                rates=rates,
                projection_rates=projection_rates,
                selectivities=selectivities,
                index_event_nodes=index_event_nodes,
                shortest_path_distances=context.shortest_path_distances,
                sink_nodes=context.sink_nodes,
                network=network,
                mode=mode,
            )
            total_baseline_cost += baseline_cost
            total_baseline_latency = max(total_baseline_latency, baseline_latency)

        # Update context with baseline (create new context as it's frozen)
        context = PlacementContext(
            network_graph=context.network_graph,
            shortest_path_distances=context.shortest_path_distances,
            sink_nodes=context.sink_nodes,
            cloud_node=context.cloud_node,
            latency_constraint=context.latency_constraint,
            baseline_latency=total_baseline_latency,
            current_placements=context.current_placements.copy(),
        )

        logger.info(
            f"Baseline: cost={total_baseline_cost:.2f}, latency={total_baseline_latency:.2f}"
        )

    # Step 2: Initial greedy placement
    initial_placements = _greedy_latency_constrained_placement(
        projections_in_order=projections_in_order,
        query_workload=query_workload,
        context=context,
        selectivities=selectivities,
        combination_dict=combination_dict,
        rates=rates,
        projection_rates=projection_rates,
        index_event_nodes=index_event_nodes,
        network_data=network_data,
        event_nodes=event_nodes,
        network=network,
        mode=mode,
    )

    if not initial_placements:
        logger.warning("Initial greedy placement failed")
        return {}

    # Step 3: Iterative refinement with cascade handling
    current_placements = initial_placements.copy()

    for iteration in range(max_iterations):
        logger.debug(f"Starting optimization iteration {iteration + 1}")

        improvements_found = False

        # Try to improve each placement considering cascade effects
        for proj in projections_in_order:
            if proj not in current_placements:
                continue

            current_decision = current_placements[proj]

            # Try alternative placements for this projection
            improved_decision = _try_improve_placement_with_cascade(
                projection=proj,
                current_decision=current_decision,
                current_placements=current_placements,
                query_workload=query_workload,
                context=context,
                selectivities=selectivities,
                combination_dict=combination_dict,
                rates=rates,
                projection_rates=projection_rates,
                index_event_nodes=index_event_nodes,
                network_data=network_data,
                event_nodes=event_nodes,
                network=network,
                mode=mode,
            )

            if improved_decision and improved_decision.costs < current_decision.costs:
                logger.debug(
                    f"Improved placement for {proj}: "
                    f"node {current_decision.node} -> {improved_decision.node}, "
                    f"cost {current_decision.costs:.2f} -> {improved_decision.costs:.2f}"
                )
                current_placements[proj] = improved_decision
                improvements_found = True

        if not improvements_found:
            logger.debug(
                f"No improvements found in iteration {iteration + 1}, terminating"
            )
            break

        logger.debug(f"Completed iteration {iteration + 1} with improvements")

    # Step 4: Validate final placements meet latency constraints
    if context.latency_constraint and context.baseline_latency:
        _validate_latency_constraints(current_placements, context)

    logger.info(
        f"Latency-aware placement completed with {len(current_placements)} placements"
    )
    return current_placements


def _greedy_latency_constrained_placement(
    projections_in_order: List[Any],
    query_workload: List[Any],
    context: PlacementContext,
    selectivities: Dict[Any, float],
    combination_dict: Dict[Any, Any],
    rates: Dict[str, float],
    projection_rates: Dict[Any, Tuple],
    index_event_nodes: Dict[str, List[str]],
    network_data: Dict[int, List[str]],
    event_nodes: List[List[int]],
    network: List[Any],
    mode: Any,
) -> Dict[Any, PlacementDecision]:
    """
    Perform initial greedy placement respecting latency constraints.

    Args:
        projections_in_order: Projections in processing order
        query_workload: Complete query workload
        context: Placement context
        selectivities: Global selectivities
        combination_dict: Dictionary containing all combinations
        rates: Global output rates
        projection_rates: Output rates and selectivities
        index_event_nodes: Mapping of primitive events to ETB indices
        network_data: Node-to-primitive-events mapping
        event_nodes: ETB emission matrix
        network: List of all node objects
        mode: Simulation mode

    Returns:
        Dict[Any, PlacementDecision]: Initial placement decisions
    """
    logger.debug("Performing greedy latency-constrained placement")

    placements = {}

    # Simulate placement state tracking
    from .initialization import initialize_placement_state

    for proj in projections_in_order:
        logger.debug(f"Processing projection {proj}")

        # Initialize placement state (simplified)
        try:
            placement_state = initialize_placement_state(
                combination=[],  # TODO: Pass actual combination if available
                proj_filter_dict={},
                no_filter=0,
                projection=proj,
                graph=context.network_graph,
            )
        except Exception as e:
            logger.warning(f"Could not initialize placement state for {proj}: {e}")
            placement_state = None

        # Check if any subqueries are already placed
        has_placed_subqueries = any(
            sub_proj in placements
            for sub_proj in projections_in_order
            if sub_proj != proj
        )

        # Get best placement for this projection
        best_decision = compute_latency_aware_placement_for_projection(
            projection=proj,
            query_workload=query_workload,
            context=context,
            selectivities=selectivities,
            combination_dict=combination_dict,
            rates=rates,
            projection_rates=projection_rates,
            index_event_nodes=index_event_nodes,
            network_data=network_data,
            event_nodes=event_nodes,
            network=network,
            mode=mode,
            has_placed_subqueries=has_placed_subqueries,
            placement_state=placement_state,
        )

        if best_decision:
            placements[proj] = best_decision
            logger.debug(
                f"Placed {proj} at node {best_decision.node} with cost {best_decision.costs:.2f}"
            )

            # Update event tracker to reflect this placement
            try:
                _update_event_availability(proj, best_decision)
            except Exception as e:
                logger.warning(f"Could not update event availability for {proj}: {e}")
        else:
            logger.warning(f"No valid placement found for {proj}")

    logger.debug(
        f"Greedy placement completed: {len(placements)}/{len(projections_in_order)} placements"
    )
    return placements


def _try_improve_placement_with_cascade(
    projection: Any,
    current_decision: PlacementDecision,
    current_placements: Dict[Any, PlacementDecision],
    query_workload: List[Any],
    context: PlacementContext,
    selectivities: Dict[Any, float],
    combination_dict: Dict[Any, Any],
    rates: Dict[str, float],
    projection_rates: Dict[Any, Tuple],
    index_event_nodes: Dict[str, List[str]],
    network_data: Dict[int, List[str]],
    event_nodes: List[List[int]],
    network: List[Any],
    mode: Any,
) -> Optional[PlacementDecision]:
    """
    Try to improve placement for a single projection considering cascade effects.

    Args:
        projection: The projection to improve
        current_decision: Current placement decision
        current_placements: All current placements
        query_workload: Complete query workload
        context: Placement context
        selectivities: Global selectivities
        combination_dict: Dictionary containing all combinations
        rates: Global output rates
        projection_rates: Output rates and selectivities
        index_event_nodes: Mapping of primitive events to ETB indices
        network_data: Node-to-primitive-events mapping
        event_nodes: ETB emission matrix
        network: List of all node objects
        mode: Simulation mode

    Returns:
        Optional[PlacementDecision]: Improved placement or None if no improvement
    """
    logger.debug(f"Trying to improve placement for {projection}")

    # Remove current placement temporarily to recalculate
    temp_placements = current_placements.copy()
    del temp_placements[projection]

    # Simulate placement state
    try:
        from .initialization import initialize_placement_state

        placement_state = initialize_placement_state(
            combination=[],  # TODO: Pass actual combination if available
            proj_filter_dict={},
            no_filter=0,
            projection=projection,
            graph=context.network_graph,
        )
    except Exception as e:
        logger.debug(f"Could not initialize placement state for {projection}: {e}")
        placement_state = None

    # Check for placed subqueries
    has_placed_subqueries = len(temp_placements) > 0

    # Try to find better placement
    improved_decision = compute_latency_aware_placement_for_projection(
        projection=projection,
        query_workload=query_workload,
        context=context,
        selectivities=selectivities,
        combination_dict=combination_dict,
        rates=rates,
        projection_rates=projection_rates,
        index_event_nodes=index_event_nodes,
        network_data=network_data,
        event_nodes=event_nodes,
        network=network,
        mode=mode,
        has_placed_subqueries=has_placed_subqueries,
        placement_state=placement_state,
    )

    if improved_decision and improved_decision.costs < current_decision.costs:
        # Verify latency constraint is still met
        if context.latency_constraint and context.baseline_latency:
            max_allowed_latency = (
                context.baseline_latency * context.latency_constraint.max_latency_factor
            )
            if improved_decision.latency > max_allowed_latency:
                logger.debug(
                    f"Improved placement for {projection} violates latency constraint"
                )
                return None

        return improved_decision

    return None


def _update_event_availability(projection: Any, decision: PlacementDecision) -> None:
    """
    Update global event availability tracker with new placement.

    Args:
        projection: The placed projection
        decision: The placement decision
    """
    try:
        from .cost_calculation import get_events_for_projection

        global_event_tracker = get_global_event_placement_tracker()
        new_events = list(get_events_for_projection(projection))

        if new_events:
            global_event_tracker.add_events_at_node(
                node_id=decision.node,
                events=new_events,
                query_id=str(projection),
                acquisition_type=decision.strategy,
                acquisition_steps=decision.plan_details.get("acquisition_steps", []),
            )

            logger.debug(
                f"Updated event availability: {len(new_events)} events at node {decision.node}"
            )

    except Exception as e:
        logger.warning(f"Could not update event availability: {e}")


def _validate_latency_constraints(
    placements: Dict[Any, PlacementDecision],
    context: PlacementContext,
) -> None:
    """
    Validate that all placements meet latency constraints.

    Args:
        placements: Final placement decisions
        context: Placement context with constraints
    """
    if not context.latency_constraint or not context.baseline_latency:
        return

    max_allowed_latency = (
        context.baseline_latency * context.latency_constraint.max_latency_factor
    )
    violations = []

    for proj, decision in placements.items():
        if decision.latency and decision.latency > max_allowed_latency:
            violations.append((proj, decision.latency, max_allowed_latency))

    if violations:
        logger.warning(
            f"Latency constraint violations found: {len(violations)} projections"
        )
        for proj, latency, max_latency in violations:
            logger.warning(
                f"  {proj}: latency={latency:.2f} > max_allowed={max_latency:.2f}"
            )
    else:
        logger.info("All placements meet latency constraints")


def handle_infeasible_latency_constraint(
    projections: List[Any],
    context: PlacementContext,
    fallback_strategy: str = "relax_constraint",
) -> PlacementContext:
    """
    Handle cases where no placement satisfies latency constraints.

    Args:
        projections: List of projections to place
        context: Current placement context
        fallback_strategy: Strategy to use ("relax_constraint", "closest_solution", "cloud_fallback")

    Returns:
        PlacementContext: Updated context with fallback handling
    """
    logger.warning(
        f"Handling infeasible latency constraint with strategy: {fallback_strategy}"
    )

    if fallback_strategy == "relax_constraint":
        # Increase latency factor by 20%
        if context.latency_constraint:
            new_factor = context.latency_constraint.max_latency_factor * 1.2
            logger.info(
                f"Relaxing latency constraint factor from {context.latency_constraint.max_latency_factor:.2f} to {new_factor:.2f}"
            )

            relaxed_constraint = LatencyConstraint(
                max_latency_factor=new_factor,
                reference_strategy=context.latency_constraint.reference_strategy,
                absolute_max_latency=context.latency_constraint.absolute_max_latency,
            )

            return PlacementContext(
                network_graph=context.network_graph,
                shortest_path_distances=context.shortest_path_distances,
                sink_nodes=context.sink_nodes,
                cloud_node=context.cloud_node,
                latency_constraint=relaxed_constraint,
                baseline_latency=context.baseline_latency,
                current_placements=context.current_placements.copy(),
            )

    elif fallback_strategy == "cloud_fallback":
        # Remove latency constraint and use cloud placement
        logger.info("Falling back to cloud placement (no latency constraint)")

        return PlacementContext(
            network_graph=context.network_graph,
            shortest_path_distances=context.shortest_path_distances,
            sink_nodes=context.sink_nodes,
            cloud_node=context.cloud_node,
            latency_constraint=None,
            baseline_latency=context.baseline_latency,
            current_placements=context.current_placements.copy(),
        )

    # Default: return original context
    return context
