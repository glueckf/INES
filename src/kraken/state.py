"""
Data structures and state management for the placement engine.

This module contains all the dataclasses and state objects used throughout
the placement engine, including placement decisions, runtime services,
and context objects.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any, Optional
import time
import threading

import networkx as nx
import numpy as np

from .cost_calculation import get_events_for_projection
from .logging import get_kraken_logger
from .node_tracker import get_global_event_placement_tracker

logger = get_kraken_logger(__name__)


@dataclass(frozen=True)
class RuntimeServices:
    """Runtime services providing deterministic random number generation and configuration."""

    rng: "np.random.Generator"
    debug: bool = False
    strict_compat: bool = False
    seed: Optional[int] = None

    @classmethod
    def create_deterministic(
        cls,
        seed: Optional[int] = None,
        debug: bool = False,
        strict_compat: bool = False,
    ) -> "RuntimeServices":
        """
        Create deterministic runtime services with seeded RNG.

        Args:
            seed: Random seed for deterministic behavior (uses default if None)
            debug: Enable debug mode with invariant checking
            strict_compat: Enable strict compatibility mode

        Returns:
            RuntimeServices instance with deterministic RNG
        """
        import os

        # Get seed from environment or use provided/default
        if seed is None:
            seed = int(os.environ.get("PLACEMENT_SEED", 12345))

        # Get debug/strict flags from environment if not provided
        debug = debug or os.environ.get("PLACEMENT_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )
        strict_compat = strict_compat or os.environ.get(
            "PLACEMENT_STRICT_COMPAT", ""
        ).lower() in ("1", "true", "yes")

        rng = np.random.default_rng(seed)

        return cls(rng=rng, debug=debug, strict_compat=strict_compat, seed=seed)


@dataclass(frozen=True)
class CostingPolicy:
    """Policy parameters for cost calculation and fallback strategies."""

    fallback_factor: float = 0.8
    min_savings: float = 0.0  # require improvement to switch strategies


@dataclass(frozen=True)
class SubgraphBundle:
    """Bundle containing all subgraph-related data structures."""

    subgraph: "nx.Graph"
    node_mapping: Dict[int, int]
    reverse_mapping: Dict[int, int]
    event_nodes_sub: List[List[int]]
    index_event_nodes_sub: Dict[str, List[str]]
    network_data_sub: Dict[int, List[str]]
    all_pairs_sub: List[List[float]]
    relevant_nodes: Set[int]
    placement_node_sub: int
    sub_network: List[Any]


@dataclass(frozen=True)
class ResourceReport:
    """Report on resource availability for a given node."""

    ok: bool
    reasons: Tuple[str, ...] = ()


class PlacementDecision:
    """
    Represents a placement decision for a projection at a specific node.

    This class encapsulates all information about placing a projection at a particular
    network node, including costs, strategy used, resource constraints, and metadata.
    """

    def __init__(
        self,
        node: int,
        costs: float,
        strategy: str,
        all_push_costs: float,
        push_pull_costs: Optional[float] = None,
        has_sufficient_resources: bool = False,
        plan_details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a placement decision.

        Args:
            node: The network node where the projection is placed
            costs: The final cost of this placement decision
            strategy: The strategy used ('all_push' or 'push_pull')
            all_push_costs: The cost if using all-push strategy
            push_pull_costs: The cost if using push-pull strategy (None if not calculated)
            has_sufficient_resources: Whether the node has sufficient resources for push-pull
            plan_details: Details about the evaluation plan generated
        """
        self.node = node
        self.costs = costs
        self.strategy = strategy
        self.all_push_costs = all_push_costs
        self.push_pull_costs = push_pull_costs
        self.has_sufficient_resources = has_sufficient_resources
        self.plan_details = plan_details or {}
        self.savings = all_push_costs - costs if costs < all_push_costs else 0.0

    def __str__(self) -> str:
        return f"PlacementDecision(node={self.node}, strategy={self.strategy}, costs={self.costs:.2f}, savings={self.savings:.2f})"

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization or logging."""
        return {
            "node": self.node,
            "costs": self.costs,
            "strategy": self.strategy,
            "all_push_costs": self.all_push_costs,
            "push_pull_costs": self.push_pull_costs,
            "has_sufficient_resources": self.has_sufficient_resources,
            "plan_details": self.plan_details,
            "savings": self.savings,
        }


class PlacementDecisionTracker:
    """
    Tracks and manages placement decisions for a projection across multiple nodes.

    This class maintains a collection of placement decisions and provides utilities
    for analyzing and selecting the best placement option.
    """

    def __init__(self, current_projection):
        """
        Initialize the decision tracker.

        Args:
            current_projection: The projection object being tracked
        """
        self.projection = current_projection
        self.decisions: List[PlacementDecision] = []
        self.best_decision: Optional[PlacementDecision] = None

    def add_decision(self, decision: PlacementDecision) -> None:
        """
        Add a new placement decision and update the best decision if applicable.

        Args:
            decision: The placement decision to add
        """
        self.decisions.append(decision)

        # Update best decision if this is better
        if self.best_decision is None or decision.costs < self.best_decision.costs:
            self.best_decision = decision

    def get_best_decision(self) -> Optional[PlacementDecision]:
        """Get the best (lowest cost) placement decision."""
        return self.best_decision

    def get_decisions_by_strategy(self, strategy: str) -> List[PlacementDecision]:
        """Get all decisions using a specific strategy."""
        return [d for d in self.decisions if d.strategy == strategy]

    def get_resource_constrained_decisions(self) -> List[PlacementDecision]:
        """Get decisions where resources were insufficient for push-pull."""
        return [d for d in self.decisions if not d.has_sufficient_resources]

    def summarize(self) -> str:
        """Generate a summary of all placement decisions."""
        if not self.decisions:
            return "No placement decisions recorded."

        summary = []
        summary.append(f"Placement Analysis for {self.projection}:")
        summary.append(f"  Total candidates evaluated: {len(self.decisions)}")

        push_pull_decisions = self.get_decisions_by_strategy("push_pull")
        all_push_decisions = self.get_decisions_by_strategy("all_push")
        resource_constrained = self.get_resource_constrained_decisions()

        summary.append(f"  Push-pull viable: {len(push_pull_decisions)}")
        summary.append(f"  All-push only: {len(all_push_decisions)}")
        summary.append(f"  Resource constrained: {len(resource_constrained)}")

        if self.best_decision:
            summary.append(f"  Best placement: Node {self.best_decision.node}")
            summary.append(f"  Best strategy: {self.best_decision.strategy}")
            summary.append(f"  Best cost: {self.best_decision.costs:.2f}")
            summary.append(f"  Savings vs all-push: {self.best_decision.savings:.2f}")

        return "\n".join(summary)

    def export_decisions(self) -> List[Dict[str, Any]]:
        """Export all decisions as a list of dictionaries."""
        return [decision.to_dict() for decision in self.decisions]


def check_if_projection_has_placed_subqueries(
    current_projection, dependencies_per_projection, global_tracker
) -> bool:
    """
    Check if a projection has subqueries that have already been placed.

    Args:
        current_projection: The projection to check
        dependencies_per_projection: Combination dictionary mapping projections to subqueries
        global_tracker: Global placement tracker instance

    Returns:
        bool: True if any subqueries have existing placements

    Raises:
        ValueError: If projection not found in mycombi or invalid structure
    """
    try:
        subqueries = dependencies_per_projection[current_projection]
    except KeyError as e:
        raise ValueError("Projection not found in mycombi or invalid structure") from e

    global_tracker.register_query_hierarchy(current_projection, subqueries)

    placed = global_tracker.placed_subqueries_set()  # O(1) membership
    return any((hasattr(sq, "children") and (sq in placed)) for sq in subqueries)


def update_tracker(best_decision, placement_decision_tracker, projection) -> None:
    """
    Update global placement and event trackers with the best placement decision.

    Args:
        best_decision: The best placement decision selected
        placement_decision_tracker: Tracker containing all placement decisions
        projection: The projection that was placed

    Raises:
        RuntimeError: If no valid placement nodes found
    """
    if not best_decision:
        raise RuntimeError("No valid placement nodes found")

    from .global_placement_tracker import get_global_placement_tracker

    global_placement_tracker = get_global_placement_tracker()
    global_event_placement_tracker = get_global_event_placement_tracker()

    global_placement_tracker.store_placement_decisions(
        projection, placement_decision_tracker
    )

    new_events_available = list(get_events_for_projection(projection))

    global_event_placement_tracker.add_events_at_node(
        node_id=best_decision.node,
        events=new_events_available,
        query_id=str(projection),
        acquisition_type=best_decision.strategy,
        acquisition_steps=best_decision.plan_details.get("aquisition_steps", []),
    )


class KrakenTimingTracker:
    """
    Thread-safe timing tracker for Kraken placement engine operations.

    Tracks detailed timing information for placement calculations and prepp operations
    to provide granular performance metrics.
    """

    def __init__(self):
        """Initialize the timing tracker with thread-safe counters."""
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Reset all timing counters to zero."""
        with self._lock:
            self.total_prepp_time = 0.0
            self.prepp_call_count = 0
            self.placement_start_time = None
            self.total_placement_time = 0.0
            self.placement_evaluations_count = 0

    def start_placement_timing(self) -> None:
        """Start timing the overall placement computation."""
        with self._lock:
            self.placement_start_time = time.time()

    def add_prepp_time(self, duration: float) -> None:
        """
        Add time spent in prepp computation.

        Args:
            duration: Time in seconds spent in prepp computation
        """
        with self._lock:
            self.total_prepp_time += duration
            self.prepp_call_count += 1

    def increment_placement_evaluations(self) -> None:
        """Increment the count of placement evaluations (nodes considered for placement)."""
        with self._lock:
            self.placement_evaluations_count += 1

    def finalize_placement_timing(self) -> Tuple[float, float, float, int, int]:
        """
        Finalize placement timing and calculate metrics.

        Returns:
            Tuple of (total_time, placement_only_time, prepp_total_time, prepp_call_count, placement_evaluations_count)
        """
        with self._lock:
            if self.placement_start_time is None:
                logger.warning("Placement timing was not started properly")
                return 0.0, 0.0, 0.0, 0, 0

            end_time = time.time()
            total_time = end_time - self.placement_start_time
            placement_only_time = total_time - self.total_prepp_time

            logger.debug(
                f"Timing breakdown: total={total_time:.3f}s, "
                f"placement_only={placement_only_time:.3f}s, "
                f"prepp_total={self.total_prepp_time:.3f}s, "
                f"prepp_calls={self.prepp_call_count}, "
                f"evaluations={self.placement_evaluations_count}"
            )

            return (
                total_time,
                placement_only_time,
                self.total_prepp_time,
                self.prepp_call_count,
                self.placement_evaluations_count,
            )


# Global timing tracker instance
_timing_tracker = KrakenTimingTracker()


def get_kraken_timing_tracker() -> KrakenTimingTracker:
    """Get the global Kraken timing tracker instance."""
    return _timing_tracker


def reset_kraken_timing_tracker() -> None:
    """Reset the global timing tracker for a new placement computation."""
    _timing_tracker.reset()
