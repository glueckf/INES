"""
Fallback cost calculation and policies.

This module provides fallback cost calculation when push-pull
strategies fail or encounter errors.
"""

from typing import Optional
from .logging import get_kraken_logger
from .state import CostingPolicy

logger = get_kraken_logger(__name__)

# Default fallback policy constants
DEFAULT_FALLBACK_FACTOR = 0.8
DEFAULT_MIN_SAVINGS = 0.0


def should_use_push_pull(
    all_push_cost: float, push_pull_cost: float, policy: Optional[CostingPolicy] = None
) -> bool:
    """
    Determine if push-pull strategy should be used based on policy.

    Args:
        all_push_cost: Cost of all-push strategy
        push_pull_cost: Cost of push-pull strategy
        policy: Costing policy (uses defaults if None)

    Returns:
        bool: True if push-pull should be used
    """

    # TODO: Handle default values more gracefully
    if policy is None:
        policy = CostingPolicy(DEFAULT_FALLBACK_FACTOR, DEFAULT_MIN_SAVINGS)

    savings = all_push_cost - push_pull_cost
    should_use = savings >= policy.min_savings and push_pull_cost < all_push_cost

    logger.debug(
        f"Push-pull decision: savings={savings:.2f}, min_required={policy.min_savings:.2f}, use_push_pull={should_use}"
    )

    return should_use


def get_strategy_recommendation(
    all_push_costs: float,
    push_pull_costs: Optional[float],
    has_enough_resources: bool,
    policy: Optional[CostingPolicy] = None,
) -> str:
    """
    Get recommended strategy based on costs, resources, and policy.

    Args:
        all_push_costs: Cost of all-push strategy
        push_pull_costs: Cost of push-pull strategy (None if not calculated/available)
        has_enough_resources: Whether the node has sufficient resources for push-pull
        policy: Costing policy (uses defaults if None)

    Returns:
        str: Recommended strategy ('all_push' or 'push_pull')
    """
    if policy is None:
        policy = CostingPolicy(DEFAULT_FALLBACK_FACTOR, DEFAULT_MIN_SAVINGS)

    # Can't use push-pull without resources
    if not has_enough_resources:
        logger.debug("Recommending all-push: insufficient resources for push-pull")
        return "all_push"

    # Can't use push-pull if cost wasn't calculated
    if push_pull_costs is None:
        logger.debug("Recommending all-push: push-pull cost unavailable")
        return "all_push"

    # Use policy to decide
    if should_use_push_pull(all_push_costs, push_pull_costs, policy):
        logger.debug("Recommending push-pull: meets savings threshold")
        return "push_pull"
    else:
        logger.debug("Recommending all-push: insufficient savings")
        return "all_push"


def get_strategy_recommendation_with_latency_constraint(
    all_push_costs: float,
    push_pull_costs: Optional[float],
    all_push_latency: float,
    push_pull_latency: Optional[float],
    max_allowed_latency: float,
    has_enough_resources: bool,
    policy: Optional[CostingPolicy] = None,
    latency_weight: float = 0.3,
) -> str:
    """
    Get strategy recommendation considering both cost and latency constraints.

    This function extends the standard strategy recommendation with latency
    awareness, enabling multi-objective optimization between cost and latency.

    Args:
        all_push_costs: Cost of all-push strategy
        push_pull_costs: Cost of push-pull strategy (None if not calculated/available)
        all_push_latency: Latency of all-push strategy
        push_pull_latency: Latency of push-pull strategy (None if not calculated/available)
        max_allowed_latency: Maximum allowed latency threshold
        has_enough_resources: Whether the node has sufficient resources for push-pull
        policy: Costing policy (uses defaults if None)
        latency_weight: Weight for latency in multi-objective optimization (0.0-1.0)

    Returns:
        str: Recommended strategy ('all_push' or 'push_pull')
    """
    if policy is None:
        policy = CostingPolicy(DEFAULT_FALLBACK_FACTOR, DEFAULT_MIN_SAVINGS)

    # First filter by latency constraint
    valid_strategies = []

    if all_push_latency <= max_allowed_latency:
        valid_strategies.append(("all_push", all_push_costs, all_push_latency))
        logger.debug(
            f"All-push meets latency constraint: {all_push_latency:.2f} <= {max_allowed_latency:.2f}"
        )

    if (
        push_pull_costs is not None
        and push_pull_latency is not None
        and push_pull_latency <= max_allowed_latency
        and has_enough_resources
    ):
        valid_strategies.append(("push_pull", push_pull_costs, push_pull_latency))
        logger.debug(
            f"Push-pull meets latency constraint: {push_pull_latency:.2f} <= {max_allowed_latency:.2f}"
        )

    # If no strategies meet latency constraint, select closest to constraint
    if not valid_strategies:
        logger.warning("No strategies meet latency constraint, selecting closest")

        all_push_violation = all_push_latency - max_allowed_latency
        strategies_by_violation = [("all_push", all_push_violation)]

        if push_pull_latency is not None and has_enough_resources:
            push_pull_violation = push_pull_latency - max_allowed_latency
            strategies_by_violation.append(("push_pull", push_pull_violation))

        # Return strategy with smallest latency violation
        best_strategy = min(strategies_by_violation, key=lambda x: x[1])[0]
        logger.debug(
            f"Selected strategy {best_strategy} with minimal latency violation"
        )
        return best_strategy

    # Among valid strategies, use multi-objective selection if multiple options
    if len(valid_strategies) == 1:
        selected_strategy = valid_strategies[0][0]
        logger.debug(f"Only one strategy meets latency constraint: {selected_strategy}")
        return selected_strategy

    # Multi-objective optimization between cost and latency
    logger.debug(f"Multi-objective optimization with latency_weight={latency_weight}")

    # Normalize costs and latencies for comparison
    costs = [s[1] for s in valid_strategies]
    latencies = [s[2] for s in valid_strategies]

    max_cost = max(costs)
    min_cost = min(costs)
    cost_range = max_cost - min_cost if max_cost > min_cost else 1.0

    max_latency = max(latencies)
    min_latency = min(latencies)
    latency_range = max_latency - min_latency if max_latency > min_latency else 1.0

    scores = []
    for strategy, cost, latency in valid_strategies:
        # Normalize to [0, 1] where 0 is best
        normalized_cost = (cost - min_cost) / cost_range if cost_range > 0 else 0.0
        normalized_latency = (
            (latency - min_latency) / latency_range if latency_range > 0 else 0.0
        )

        # Weighted combination: lower is better
        score = (
            1 - latency_weight
        ) * normalized_cost + latency_weight * normalized_latency
        scores.append((score, strategy))

        logger.debug(
            f"Strategy {strategy}: cost={cost:.2f} (norm={normalized_cost:.3f}), "
            f"latency={latency:.2f} (norm={normalized_latency:.3f}), score={score:.3f}"
        )

    best_strategy = min(scores, key=lambda x: x[0])[1]
    logger.debug(f"Multi-objective selection chose: {best_strategy}")

    return best_strategy
