"""
Fallback cost calculation and policies.

This module provides fallback cost calculation when push-pull
strategies fail or encounter errors.
"""

from typing import Dict, Any, Optional
from .logging import get_kraken_logger
from .state import CostingPolicy

logger = get_kraken_logger(__name__)

# Default fallback policy constants
DEFAULT_FALLBACK_FACTOR = 0.8
DEFAULT_MIN_SAVINGS = 0.0


def should_use_push_pull(all_push_cost: float, push_pull_cost: float, 
                        policy: Optional[CostingPolicy] = None) -> bool:
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
    
    logger.debug(f"Push-pull decision: savings={savings:.2f}, min_required={policy.min_savings:.2f}, use_push_pull={should_use}")
    
    return should_use


def get_strategy_recommendation(all_push_costs: float, push_pull_costs: Optional[float],
                                has_enough_resources: bool, policy: Optional[CostingPolicy] = None) -> str:
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
        return 'all_push'
    
    # Can't use push-pull if cost wasn't calculated
    if push_pull_costs is None:
        logger.debug("Recommending all-push: push-pull cost unavailable")
        return 'all_push'
    
    # Use policy to decide
    if should_use_push_pull(all_push_costs, push_pull_costs, policy):
        logger.debug("Recommending push-pull: meets savings threshold")
        return 'push_pull'
    else:
        logger.debug("Recommending all-push: insufficient savings")
        return 'all_push'
