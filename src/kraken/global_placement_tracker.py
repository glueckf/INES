"""
Global placement tracker for managing hierarchical query placement decisions.

This module provides a global tracker that maintains placement decisions across
the hierarchical query decomposition process, enabling reuse of subquery placements
for complex queries like AND(A, B, D) -> AND(D, AND(A, B)).
"""

from typing import Dict, Optional, List, Any, Union
from .state import PlacementDecision, PlacementDecisionTracker
import copy


class GlobalPlacementTracker:
    """
    Global tracker that stores placement decisions for all processed projections.

    This enables hierarchical query processing where subqueries can reuse
    previously computed placement decisions with global parent placement optimization.
    """

    def __init__(self):
        """Initialize the global placement tracker."""
        self._placement_history: Dict[Any, PlacementDecisionTracker] = {}
        self._projection_mappings: Dict[Any, str] = {}
        self._query_hierarchy: Dict[Any, List[Any]] = {}
        self._subquery_to_parents: Dict[
            Any, set
        ] = {}  # Reverse mapping for O(1) lookup
        self._locked_placements: Dict[Any, bool] = {}  # Track locked subprojections
        self._parent_processing_order: List[
            Any
        ] = []  # FIFO order for conflict resolution

    def store_placement_decisions(
        self, projection: Any, placement_decisions: PlacementDecisionTracker
    ) -> None:
        """
        Store placement decisions for a projection.

        Args:
            projection: The projection object (e.g., AND(A, B))
            placement_decisions: The PlacementDecisionTracker with all decisions
        """
        self._placement_history[projection] = placement_decisions

    def get_placement_decisions(
        self, projection: Any
    ) -> Optional[PlacementDecisionTracker]:
        """
        Get stored placement decisions for a projection.

        Args:
            projection: The projection to look up

        Returns:
            PlacementDecisionTracker if found, None otherwise
        """
        return self._placement_history.get(projection)

    def get_best_placement(self, projection: Any) -> Optional[PlacementDecision]:
        """
        Get the best placement decision for a projection.

        Args:
            projection: The projection to look up

        Returns:
            PlacementDecision if found, None otherwise
        """
        tracker = self.get_placement_decisions(projection)
        if tracker:
            return tracker.get_best_decision()
        return None

    def register_query_hierarchy(
        self, parent_query: Any, subqueries: List[Any]
    ) -> None:
        """
        Register the hierarchical relationship between a query and its subqueries.

        Args:
            parent_query: The parent query (e.g., AND(A, B, D))
            subqueries: List of subqueries (e.g., ['D', AND(A, B)])
        """
        self._query_hierarchy[parent_query] = subqueries

        # Update reverse mapping for O(1) subquery parent lookups
        for subquery in subqueries:
            if subquery not in self._subquery_to_parents:
                self._subquery_to_parents[subquery] = set()
            self._subquery_to_parents[subquery].add(parent_query)

    def get_query_hierarchy(self, query: Any) -> Optional[List[Any]]:
        """
        Get the subqueries for a given query.

        Args:
            query: The query to look up

        Returns:
            List of subqueries if found, None otherwise
        """
        return self._query_hierarchy.get(query)

    def create_virtual_event_mapping(
        self, subquery: Any, virtual_name: str, placement_node: int, rate: float
    ) -> None:
        """
        Create a mapping for treating a subquery as a virtual event.

        Args:
            subquery: The original subquery (e.g., AND(A, B))
            virtual_name: Virtual event name (e.g., 'q1')
            placement_node: Node where the subquery is placed
            rate: Rate at which the virtual event is produced
        """
        self._projection_mappings[subquery] = {
            "virtual_name": virtual_name,
            "placement_node": placement_node,
            "rate": rate,
        }

    def get_virtual_event_mapping(
        self, subquery: Any
    ) -> Optional[Dict[str, Union[str, int, float]]]:
        """
        Get the virtual event mapping for a subquery.

        Args:
            subquery: The subquery to look up

        Returns:
            Dictionary with virtual event details if found, None otherwise
        """
        return self._projection_mappings.get(subquery)

    def has_placement_for(self, projection: Any) -> bool:
        """
        Check if we have placement decisions for a given projection.

        Args:
            projection: The projection to check

        Returns:
            True if placement decisions exist, False otherwise
        """
        return projection in self._placement_history

    def placed_subqueries_set(self) -> set:
        """
        Get a set of all placed subqueries for O(1) membership testing.

        Returns:
            Set of all projections that have placement decisions
        """
        return set(self._placement_history.keys())

    def update_network_structures_for_subquery(
        self,
        subquery: Any,
        parent_query: Any,
        network_data: Dict,
        rates: Dict,
        projrates: Dict,
        index_event_nodes: Dict,
    ) -> Dict[str, Any]:
        """
        Update network structures to treat a subquery as a virtual event.

        This method creates updated network structures where the subquery
        appears as a new primitive event produced by its placement node.

        Args:
            subquery: The subquery to virtualize (e.g., AND(A, B))
            parent_query: The parent query (e.g., AND(A, B, D))
            network_data: Current network data structure
            rates: Current rates dictionary
            projrates: Current projection rates
            index_event_nodes: Current index event nodes

        Returns:
            Dictionary with updated network structures
        """
        placement_decision = self.get_best_placement(subquery)
        if not placement_decision:
            raise ValueError(f"No placement decision found for subquery: {subquery}")

        # Create virtual event name
        virtual_name = f"q{len(self._projection_mappings) + 1}"

        # Get subquery rate
        subquery_rate = projrates.get(subquery, 1.0)

        # Store the mapping
        self.create_virtual_event_mapping(
            subquery, virtual_name, placement_decision.node, subquery_rate
        )

        # Create updated network structures
        updated_network_data = copy.deepcopy(network_data)
        updated_rates = copy.deepcopy(rates)
        updated_projrates = copy.deepcopy(projrates)
        updated_index_event_nodes = copy.deepcopy(index_event_nodes)

        # Add virtual event to the placement node
        placement_node = placement_decision.node
        if placement_node not in updated_network_data:
            updated_network_data[placement_node] = []

        # Add virtual event to network data
        if virtual_name not in updated_network_data[placement_node]:
            updated_network_data[placement_node].append(virtual_name)

        # Add virtual event rate
        updated_rates[virtual_name] = subquery_rate

        # Update index_event_nodes to include virtual event
        if virtual_name not in updated_index_event_nodes:
            updated_index_event_nodes[virtual_name] = []

        # Add placement node to the virtual event's node list
        if placement_node not in updated_index_event_nodes[virtual_name]:
            updated_index_event_nodes[virtual_name].append(placement_node)

        return {
            "network_data": updated_network_data,
            "rates": updated_rates,
            "projrates": updated_projrates,
            "index_event_nodes": updated_index_event_nodes,
            "virtual_name": virtual_name,
            "placement_node": placement_node,
        }

    def is_subprojection_locked(self, subprojection: Any) -> bool:
        """
        Check if a subprojection is locked by a previously processed parent.

        Args:
            subprojection: The subprojection to check

        Returns:
            True if the subprojection is locked, False otherwise
        """
        return self._locked_placements.get(subprojection, False)

    def lock_subprojection_placement(self, subprojection: Any) -> None:
        """
        Lock a subprojection to prevent future parent optimizations.

        Args:
            subprojection: The subprojection to lock
        """
        self._locked_placements[subprojection] = True

    def register_parent_processing(self, parent_projection: Any) -> None:
        """
        Register a parent projection in the FIFO processing order.

        Args:
            parent_projection: The parent projection being processed
        """
        if parent_projection not in self._parent_processing_order:
            self._parent_processing_order.append(parent_projection)

    def is_first_parent_for_subprojection(
        self, parent_projection: Any, subprojection: Any
    ) -> bool:
        """
        Check if this parent is the first to request optimization of a subprojection.

        Args:
            parent_projection: The parent projection
            subprojection: The subprojection in question

        Returns:
            True if this is the first parent, False otherwise
        """
        # Check if subprojection is already locked
        if self.is_subprojection_locked(subprojection):
            return False

        # Find all parents that use this subprojection using O(1) lookup
        parents_using_subprojection = list(
            self._subquery_to_parents.get(subprojection, set())
        )

        if not parents_using_subprojection:
            return True

        # Return True if this parent was registered first
        earliest_parent = min(
            parents_using_subprojection,
            key=lambda p: self._parent_processing_order.index(p)
            if p in self._parent_processing_order
            else float("inf"),
        )
        return earliest_parent == parent_projection

    def get_alternative_placements(
        self,
        subprojection: Any,
        parent_node: int,
        shortest_path_distances: Dict[int, Dict[int, int]],
        projection_rates: Dict[Any, tuple],
    ) -> List[Dict[str, Any]]:
        """
        Get alternative placement options for a subprojection considering parent context.

        Args:
            subprojection: The subprojection to get alternatives for
            parent_node: The node where parent projection will be placed
            shortest_path_distances: Distance matrix for calculating transfer costs
            projection_rates: Output rates for projections

        Returns:
            List of placement alternatives with total costs including transfer to parent
        """
        tracker = self.get_placement_decisions(subprojection)
        if not tracker:
            return []

        alternatives = []
        subprojection_rate = projection_rates.get(subprojection, (1.0, 1.0))[
            1
        ]  # Output rate

        for decision in tracker.decisions:
            # Calculate transfer cost from subprojection node to parent node
            transfer_distance = shortest_path_distances[decision.node][parent_node]
            transfer_cost = transfer_distance * subprojection_rate

            # Total cost = placement cost + transfer cost
            total_cost = decision.costs + transfer_cost

            alternatives.append(
                {
                    "decision": decision,
                    "transfer_cost": transfer_cost,
                    "total_cost": total_cost,
                    "placement_node": decision.node,
                    "strategy": decision.strategy,
                }
            )

        # Sort by total cost
        alternatives.sort(key=lambda x: x["total_cost"])
        return alternatives

    def optimize_subprojection_placement(
        self,
        parent_projection: Any,
        parent_node: int,
        pairwise_distance_matrix: list,
        projection_rates_selectivity: Dict[Any, tuple],
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Optimize subprojection placements for a parent projection at a specific node.

        Args:
            parent_projection: The parent projection
            parent_node: The node where parent will be placed
            pairwise_distance_matrix: Distance matrix
            projection_rates_selectivity: Output rates for projections

        Returns:
            Dictionary mapping subprojections to their optimal placement details
        """
        subqueries = self._query_hierarchy.get(parent_projection, [])
        optimized_placements = {}

        for subquery in subqueries:
            # Skip primitive events (they don't have placements)
            if not hasattr(subquery, "children"):
                continue

            # Skip locked subprojections
            if self.is_subprojection_locked(subquery):
                # Use existing placement - calculate transfer cost for parent costing
                best_placement = self.get_best_placement(subquery)
                if best_placement:
                    subprojection_rate = projection_rates_selectivity.get(
                        subquery, (1.0, 1.0)
                    )[1]
                    transfer_distance = pairwise_distance_matrix[best_placement.node][
                        parent_node
                    ]
                    transfer_cost = transfer_distance * subprojection_rate

                    optimized_placements[subquery] = {
                        "original_decision": best_placement,
                        "transfer_cost": transfer_cost,
                        "total_cost": best_placement.costs + transfer_cost,
                        "placement_node": best_placement.node,
                        "strategy": best_placement.strategy,
                        "locked": True,
                        "placement_changed": False,  # No change in placement
                    }
                continue

            # Get alternative placements considering parent context
            alternatives = self.get_alternative_placements(
                subquery,
                parent_node,
                pairwise_distance_matrix,
                projection_rates_selectivity,
            )

            if alternatives:
                # Get current best placement for comparison
                current_best = self.get_best_placement(subquery)
                best_alternative = alternatives[0]

                # Check if we're changing the placement
                placement_changed = (
                    current_best is None
                    or current_best.node != best_alternative["placement_node"]
                )

                optimized_placements[subquery] = {
                    "original_decision": current_best,
                    "optimized_decision": best_alternative["decision"],
                    "transfer_cost": best_alternative["transfer_cost"],
                    "total_cost": best_alternative["total_cost"],
                    "placement_node": best_alternative["placement_node"],
                    "strategy": best_alternative["strategy"],
                    "locked": False,
                    "placement_changed": placement_changed,
                }

        return optimized_placements

    def get_summary(self) -> str:
        """
        Get a summary of all stored placement decisions.

        Returns:
            String summary of the tracker contents
        """
        summary = ["Global Placement Tracker Summary:"]
        summary.append(f"Total projections tracked: {len(self._placement_history)}")
        summary.append(f"Query hierarchies registered: {len(self._query_hierarchy)}")
        summary.append(f"Virtual mappings created: {len(self._projection_mappings)}")
        summary.append(f"Locked placements: {len(self._locked_placements)}")

        if self._placement_history:
            summary.append("\nStored Placement Decisions:")
            for projection, tracker in self._placement_history.items():
                best = tracker.get_best_decision()
                locked = self._locked_placements.get(projection, False)
                if best:
                    lock_status = " [LOCKED]" if locked else ""
                    summary.append(
                        f"  {projection}: Node {best.node}, Strategy {best.strategy}, Cost {best.costs:.2f}{lock_status}"
                    )

        if self._query_hierarchy:
            summary.append("\nQuery Hierarchies:")
            for parent, children in self._query_hierarchy.items():
                summary.append(f"  {parent}: {children}")

        if self._parent_processing_order:
            summary.append(
                f"\nParent Processing Order: {self._parent_processing_order}"
            )

        return "\n".join(summary)


# Global instance
global_placement_tracker = GlobalPlacementTracker()


def get_global_placement_tracker() -> GlobalPlacementTracker:
    """
    Get the global placement tracker instance.

    Returns:
        The singleton GlobalPlacementTracker instance
    """
    return global_placement_tracker


def reset_global_placement_tracker() -> None:
    """
    Reset the global placement tracker (useful for testing and new workloads).
    """
    global global_placement_tracker
    global_placement_tracker = GlobalPlacementTracker()


def reset_locking_state() -> None:
    """
    Reset only the locking state while preserving placement history.
    Useful when starting a new workload that should reuse existing placements
    but allow re-optimization.
    """
    global_placement_tracker._locked_placements.clear()
    global_placement_tracker._parent_processing_order.clear()
    global_placement_tracker._subquery_to_parents.clear()
