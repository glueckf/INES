"""
Fallback cost calculation and policies.

This module provides fallback cost calculation when push-pull
strategies fail or encounter errors.
"""

from typing import Dict, List, Any, Optional
from .logging import get_placement_logger
from .state import CostingPolicy

logger = get_placement_logger(__name__)

# Default fallback policy constants
DEFAULT_FALLBACK_FACTOR = 0.8
DEFAULT_MIN_SAVINGS = 0.0


def calculate_fallback_costs(node: int, subgraph: Dict[str, Any], projection: Any, 
                           all_push_baseline: Optional[float] = None, 
                           policy: Optional[CostingPolicy] = None) -> float:
    """
    Fallback cost calculation if prePP fails.
    Uses all-push baseline if available, otherwise estimates cost.
    
    Args:
        node: Placement node
        subgraph: Subgraph information
        projection: Projection being placed
        all_push_baseline: Pre-calculated all-push costs
        policy: Costing policy (uses defaults if None)
        
    Returns:
        float: Fallback cost estimate
    """
    if policy is None:
        policy = CostingPolicy(DEFAULT_FALLBACK_FACTOR, DEFAULT_MIN_SAVINGS)
    
    try:
        logger.warning(f"Calculating fallback costs for node {node}")
        
        # If we have all-push baseline, use a conservative estimate
        # (assume push-pull is at most X% better than all-push)
        if all_push_baseline is not None:
            fallback_cost = all_push_baseline * policy.fallback_factor
            logger.info(f"Using conservative estimate: {fallback_cost:.2f} ({policy.fallback_factor*100}% of all-push baseline)")
            return fallback_cost
        
        # Otherwise, simple cost based on distance from sources to sink
        total_cost = 0.0
        all_pairs = subgraph['all_pairs_sub']
        remapped_node = subgraph['placement_node_remapped']

        # Calculate basic routing costs
        for i in range(len(all_pairs)):
            if i != remapped_node:
                total_cost += all_pairs[remapped_node][i]

        logger.info(f"Basic routing costs calculated: {total_cost:.2f}")
        return total_cost

    except Exception as e:
        logger.error(f"Error in fallback cost calculation: {e}")
        return float('inf')


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
    if policy is None:
        policy = CostingPolicy(DEFAULT_FALLBACK_FACTOR, DEFAULT_MIN_SAVINGS)
    
    savings = all_push_cost - push_pull_cost
    should_use = savings >= policy.min_savings and push_pull_cost < all_push_cost
    
    logger.debug(f"Push-pull decision: savings={savings:.2f}, min_required={policy.min_savings:.2f}, use_push_pull={should_use}")
    
    return should_use


def get_strategy_recommendation(all_push_cost: float, push_pull_cost: Optional[float], 
                               has_resources: bool, policy: Optional[CostingPolicy] = None) -> str:
    """
    Get recommended strategy based on costs, resources, and policy.
    
    Args:
        all_push_cost: Cost of all-push strategy
        push_pull_cost: Cost of push-pull strategy (None if not calculated/available)
        has_resources: Whether the node has sufficient resources for push-pull
        policy: Costing policy (uses defaults if None)
        
    Returns:
        str: Recommended strategy ('all_push' or 'push_pull')
    """
    if policy is None:
        policy = CostingPolicy(DEFAULT_FALLBACK_FACTOR, DEFAULT_MIN_SAVINGS)
    
    # Can't use push-pull without resources
    if not has_resources:
        logger.debug("Recommending all-push: insufficient resources for push-pull")
        return 'all_push'
    
    # Can't use push-pull if cost wasn't calculated
    if push_pull_cost is None:
        logger.debug("Recommending all-push: push-pull cost unavailable")
        return 'all_push'
    
    # Use policy to decide
    if should_use_push_pull(all_push_cost, push_pull_cost, policy):
        logger.debug("Recommending push-pull: meets savings threshold")
        return 'push_pull'
    else:
        logger.debug("Recommending all-push: insufficient savings")
        return 'all_push'