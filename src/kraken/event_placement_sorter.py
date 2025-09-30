from typing import List, Any, Dict, Optional


class EventPlacementSorter:
    """
    Handles sorting of candidate nodes based on event placement tracking.
    Easy to swap out for different tracking implementations.
    """

    def __init__(
        self,
        event_tracker=None,
        event_stack: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        """Initialize with either event_tracker (legacy) or event_stack (new).

        Args:
            event_tracker: Legacy global event tracker (deprecated)
            event_stack: New stack-based event tracking dictionary
        """
        self.event_tracker = event_tracker
        self.event_stack = event_stack

    def sort_candidate_nodes_optimized(
        self,
        possible_placement_nodes: List[int],
        current_projection: Any,
        primitive_events_per_projection: Dict[str, List[Any]],
    ) -> List[int]:
        """
        Optimized sorting with fixed logic and better performance.
        """

        if not possible_placement_nodes:
            return []

        # Convert projection to string key
        projection_key = str(current_projection)
        if projection_key not in primitive_events_per_projection:
            return possible_placement_nodes

        current_projection_events = primitive_events_per_projection[projection_key]

        # Separate nodes into preferred and others
        preferred_nodes = []
        other_nodes = []

        for node in reversed(possible_placement_nodes):  # Prioritize last added
            try:
                events_at_node = self._get_events_at_node(node)
                if events_at_node == current_projection_events:
                    preferred_nodes.append(node)
                else:
                    other_nodes.append(node)
            except Exception:
                # If we can't get events for this node, put it in others
                other_nodes.append(node)

        # Return preferred nodes first, then others
        return preferred_nodes + other_nodes

    def _get_events_at_node(self, node: int) -> List[str]:
        """Get events at a node using either stack or tracker.

        Args:
            node: Node ID to query

        Returns:
            List of event names at the node
        """
        if self.event_stack is not None:
            # Use new stack-based approach
            return self.event_stack.get(node, {}).get("events", [])
        elif self.event_tracker is not None:
            # Use legacy tracker
            return self.event_tracker.get_events_at_node(node)
        else:
            return []
