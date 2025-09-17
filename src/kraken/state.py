"""
Data structures and state management for the placement engine.

This module contains all the dataclasses and state objects used throughout
the placement engine, including placement decisions, runtime services,
and context objects.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any, Optional

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
class LatencyConstraint:
    """
    Latency constraint for placement decisions.

    Defines maximum allowable latency relative to a reference strategy
    (typically all-push-to-cloud baseline).
    """

    max_latency_factor: float = 1.5  # e.g., 1.5 means 50% more latency than baseline
    reference_strategy: str = "all_push_cloud"  # baseline strategy for comparison
    absolute_max_latency: Optional[float] = None  # hard upper bound if specified

    def __post_init__(self) -> None:
        """Validate latency constraint parameters."""
        if self.max_latency_factor <= 0:
            raise ValueError(
                f"max_latency_factor must be positive, got {self.max_latency_factor}"
            )
        if self.absolute_max_latency is not None and self.absolute_max_latency <= 0:
            raise ValueError(
                f"absolute_max_latency must be positive, got {self.absolute_max_latency}"
            )

    def is_latency_acceptable(self, latency: float, baseline_latency: float) -> bool:
        """
        Check if a latency value meets the constraint.

        Args:
            latency: The latency to check
            baseline_latency: The baseline reference latency

        Returns:
            bool: True if latency meets constraint
        """
        relative_ok = latency <= baseline_latency * self.max_latency_factor
        absolute_ok = (
            self.absolute_max_latency is None or latency <= self.absolute_max_latency
        )
        return relative_ok and absolute_ok


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
        latency: Optional[float] = None,
        all_push_latency: Optional[float] = None,
        push_pull_latency: Optional[float] = None,
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
            latency: The final latency of this placement decision
            all_push_latency: The latency if using all-push strategy
            push_pull_latency: The latency if using push-pull strategy
        """
        self.node = node
        self.costs = costs
        self.strategy = strategy
        self.all_push_costs = all_push_costs
        self.push_pull_costs = push_pull_costs
        self.has_sufficient_resources = has_sufficient_resources
        self.plan_details = plan_details or {}
        self.latency = latency
        self.all_push_latency = all_push_latency
        self.push_pull_latency = push_pull_latency
        self.savings = all_push_costs - costs if costs < all_push_costs else 0.0

    def __str__(self) -> str:
        latency_str = (
            f", latency={self.latency:.2f}" if self.latency is not None else ""
        )
        return f"PlacementDecision(node={self.node}, strategy={self.strategy}, costs={self.costs:.2f}, savings={self.savings:.2f}{latency_str})"

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
            "latency": self.latency,
            "all_push_latency": self.all_push_latency,
            "push_pull_latency": self.push_pull_latency,
            "savings": self.savings,
        }


class PlacementDecisionTracker:
    """
    Tracks and manages placement decisions for a projection across multiple nodes.

    This class maintains a collection of placement decisions and provides utilities
    for analyzing and selecting the best placement option.
    """

    def __init__(self, projection):
        """
        Initialize the decision tracker.

        Args:
            projection: The projection object being tracked
        """
        self.projection = projection
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


@dataclass(frozen=True)
class EventAvailabilityUpdate:
    """
    Represents an update to event availability after a placement decision.

    Used for cascade effect tracking.
    """

    node_id: int
    events: List[str]
    projection: Any
    strategy: str


@dataclass(frozen=True)
class PlacementContext:
    """
    Context information for placement computation including latency constraints.
    """

    network_graph: "nx.Graph"
    shortest_path_distances: List[List[float]]
    sink_nodes: List[int]
    cloud_node: int
    latency_constraint: Optional[LatencyConstraint] = None
    baseline_latency: Optional[float] = None
    current_placements: Dict[Any, PlacementDecision] = None

    def __post_init__(self) -> None:
        """Initialize mutable fields."""
        if self.current_placements is None:
            object.__setattr__(self, "current_placements", {})


def check_if_projection_has_placed_subqueries(
    projection, mycombi, global_tracker
) -> bool:
    """
    Check if a projection has subqueries that have already been placed.

    Args:
        projection: The projection to check
        mycombi: Combination dictionary mapping projections to subqueries
        global_tracker: Global placement tracker instance

    Returns:
        bool: True if any subqueries have existing placements

    Raises:
        ValueError: If projection not found in mycombi or invalid structure
    """
    if projection in mycombi:
        subqueries = mycombi[projection]
        global_tracker.register_query_hierarchy(projection, subqueries)

        # Check if any subqueries have existing placements
        for subquery in subqueries:
            has_children = hasattr(subquery, "children")
            has_placement = global_tracker.has_placement_for(subquery)
            if has_children and has_placement:
                return True

        return False
    else:
        raise ValueError("Projection not found in mycombi or invalid structure")


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
