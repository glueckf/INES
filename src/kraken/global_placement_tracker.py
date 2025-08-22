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
    previously computed placement decisions.
    """
    
    def __init__(self):
        """Initialize the global placement tracker."""
        self._placement_history: Dict[Any, PlacementDecisionTracker] = {}
        self._projection_mappings: Dict[Any, str] = {}
        self._query_hierarchy: Dict[Any, List[Any]] = {}
        
    def store_placement_decisions(self, projection: Any, tracker: PlacementDecisionTracker) -> None:
        """
        Store placement decisions for a projection.
        
        Args:
            projection: The projection object (e.g., AND(A, B))
            tracker: The PlacementDecisionTracker with all decisions
        """
        self._placement_history[projection] = tracker
        
    def get_placement_decisions(self, projection: Any) -> Optional[PlacementDecisionTracker]:
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
        
    def register_query_hierarchy(self, parent_query: Any, subqueries: List[Any]) -> None:
        """
        Register the hierarchical relationship between a query and its subqueries.
        
        Args:
            parent_query: The parent query (e.g., AND(A, B, D))
            subqueries: List of subqueries (e.g., ['D', AND(A, B)])
        """
        self._query_hierarchy[parent_query] = subqueries
        
    def get_query_hierarchy(self, query: Any) -> Optional[List[Any]]:
        """
        Get the subqueries for a given query.
        
        Args:
            query: The query to look up
            
        Returns:
            List of subqueries if found, None otherwise
        """
        return self._query_hierarchy.get(query)
        
    def create_virtual_event_mapping(self, subquery: Any, virtual_name: str, placement_node: int, rate: float) -> None:
        """
        Create a mapping for treating a subquery as a virtual event.
        
        Args:
            subquery: The original subquery (e.g., AND(A, B))
            virtual_name: Virtual event name (e.g., 'q1')
            placement_node: Node where the subquery is placed
            rate: Rate at which the virtual event is produced
        """
        self._projection_mappings[subquery] = {
            'virtual_name': virtual_name,
            'placement_node': placement_node,
            'rate': rate
        }
        
    def get_virtual_event_mapping(self, subquery: Any) -> Optional[Dict[str, Union[str, int, float]]]:
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
        
    def update_network_structures_for_subquery(
        self, 
        subquery: Any,
        parent_query: Any,
        network_data: Dict,
        rates: Dict,
        projrates: Dict,
        index_event_nodes: Dict
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
            'network_data': updated_network_data,
            'rates': updated_rates,
            'projrates': updated_projrates,
            'index_event_nodes': updated_index_event_nodes,
            'virtual_name': virtual_name,
            'placement_node': placement_node
        }
        
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
        
        if self._placement_history:
            summary.append("\nStored Placement Decisions:")
            for projection, tracker in self._placement_history.items():
                best = tracker.get_best_decision()
                if best:
                    summary.append(f"  {projection}: Node {best.node}, Strategy {best.strategy}, Cost {best.costs:.2f}")
                    
        if self._query_hierarchy:
            summary.append("\nQuery Hierarchies:")
            for parent, children in self._query_hierarchy.items():
                summary.append(f"  {parent}: {children}")
                
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
    Reset the global placement tracker (useful for testing).
    """
    global global_placement_tracker
    global_placement_tracker = GlobalPlacementTracker()