"""Event placement tracking module for distributed query processing."""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass


@dataclass
class EventMetadata:
    """Metadata associated with an event at a node.

    Attributes:
        query_id: Identifier of the query this event belongs to.
        acquisition_type: How the event was acquired (e.g., 'generated', 'pulled').
        acquisition_steps: Dictionary containing pull information per step.
    """

    query_id: Optional[str] = None
    acquisition_type: str = "generated"
    acquisition_steps: Optional[Dict[int, Dict[str, List[str]]]] = None

    def get_query_id(self) -> Optional[str]:
        """Get the query ID associated with this event metadata."""
        return self.query_id


class EventPlacementTracker:
    """Tracks event placement and availability across distributed nodes.

    This class maintains bidirectional mapping between events and nodes,
    supporting complex placement scenarios where events can be present at
    multiple nodes and nodes can host multiple events.

    The tracker supports two primary use cases:
    1. Tracking which events are available at each node
    2. Tracking which nodes have specific events after placement decisions
    """

    def __init__(self) -> None:
        """Initialize the event placement tracker with empty mappings."""
        # Maps node_id -> list of events present at that node
        self._node_to_events: Dict[int, List[str]] = {}

        # Maps event_name -> list of nodes where this event is present
        self._event_to_nodes: Dict[str, List[int]] = {}

        # Maps (node_id, event_name) -> metadata
        self._event_metadata: Dict[tuple[int, str], EventMetadata] = {}

    def add_event_at_node(
        self,
        node_id: int,
        event: str,
        query_id: Optional[str] = None,
        acquisition_type: str = "generated",
        acquisition_steps: Optional[Dict[int, Dict[str, List[str]]]] = None,
    ) -> None:
        """Add an event to a specific node with optional metadata.

        Args:
            node_id: The node identifier where the event is placed.
            event: The event name to be tracked.
            query_id: Optional query identifier for this event.
            acquisition_type: How the event was acquired.
            acquisition_steps: Pull information per processing step.
        """
        # Add event to node mapping
        if node_id not in self._node_to_events:
            self._node_to_events[node_id] = []
        if event not in self._node_to_events[node_id]:
            self._node_to_events[node_id].append(event)

        # Add node to event mapping
        if event not in self._event_to_nodes:
            self._event_to_nodes[event] = []
        if node_id not in self._event_to_nodes[event]:
            self._event_to_nodes[event].append(node_id)

        # Store metadata
        metadata_key = (node_id, event)
        self._event_metadata[metadata_key] = EventMetadata(
            query_id=query_id,
            acquisition_type=acquisition_type,
            acquisition_steps=acquisition_steps,
        )

    def add_events_at_node(
        self,
        node_id: int,
        events: List[str],
        query_id: Optional[str] = None,
        acquisition_type: str = "placed",
        acquisition_steps: Optional[Dict[int, Dict[str, List[str]]]] = None,
    ) -> None:
        """Add multiple events to a node (e.g., after placement decision).

        Args:
            node_id: The node identifier where events are placed.
            events: List of event names to be tracked.
            query_id: Optional query identifier for these events.
            acquisition_type: How the events were acquired.
            acquisition_steps: Pull information per processing step.
        """
        for event in events:
            self.add_event_at_node(
                node_id=node_id,
                event=event,
                query_id=query_id,
                acquisition_type=acquisition_type,
                acquisition_steps=acquisition_steps,
            )

    def get_events_at_node(self, node_id: int) -> List[str]:
        """Get all events present at a specific node.

        Args:
            node_id: The node identifier to query.

        Returns:
            List of event names present at the node, empty if node not found.
        """
        return self._node_to_events.get(node_id, []).copy()

    def get_nodes_with_event(self, event: str) -> List[int]:
        """Get all nodes that have a specific event.

        Args:
            event: The event name to query.

        Returns:
            List of node identifiers that have this event, empty if not found.
        """
        return self._event_to_nodes.get(event, []).copy()

    def get_event_metadata(self, node_id: int, event: str) -> Optional[EventMetadata]:
        """Get metadata for a specific event at a specific node.

        Args:
            node_id: The node identifier.
            event: The event name.

        Returns:
            EventMetadata if found, None otherwise.
        """
        return self._event_metadata.get((node_id, event))

    def remove_event_from_node(self, node_id: int, event: str) -> bool:
        """Remove an event from a specific node.

        Args:
            node_id: The node identifier.
            event: The event name to remove.

        Returns:
            True if event was removed, False if not found.
        """
        removed = False

        # Remove from node_to_events mapping
        if node_id in self._node_to_events and event in self._node_to_events[node_id]:
            self._node_to_events[node_id].remove(event)
            if not self._node_to_events[node_id]:  # Clean up empty lists
                del self._node_to_events[node_id]
            removed = True

        # Remove from event_to_nodes mapping
        if event in self._event_to_nodes and node_id in self._event_to_nodes[event]:
            self._event_to_nodes[event].remove(node_id)
            if not self._event_to_nodes[event]:  # Clean up empty lists
                del self._event_to_nodes[event]

        # Remove metadata
        metadata_key = (node_id, event)
        if metadata_key in self._event_metadata:
            del self._event_metadata[metadata_key]

        return removed

    def get_all_nodes(self) -> Set[int]:
        """Get all node identifiers that have events.

        Returns:
            Set of node identifiers.
        """
        return set(self._node_to_events.keys())

    def get_all_events(self) -> Set[str]:
        """Get all event names being tracked.

        Returns:
            Set of event names.
        """
        return set(self._event_to_nodes.keys())

    def has_event_at_node(self, node_id: int, event: str) -> bool:
        """Check if a specific event exists at a specific node.

        Args:
            node_id: The node identifier to check.
            event: The event name to check.

        Returns:
            True if the event exists at the node, False otherwise.
        """
        return (
            node_id in self._node_to_events and event in self._node_to_events[node_id]
        )

    def clear(self) -> None:
        """Clear all tracking data."""
        self._node_to_events.clear()
        self._event_to_nodes.clear()
        self._event_metadata.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current tracking state.

        Returns:
            Dictionary containing summary statistics and mappings.
        """
        return {
            "total_nodes": len(self._node_to_events),
            "total_events": len(self._event_to_nodes),
            "total_placements": len(self._event_metadata),
            "node_to_events": dict(self._node_to_events),
            "event_to_nodes": dict(self._event_to_nodes),
        }

    def find_aquisition_step_for_event_at_node(self, node_id: int, event: str):
        metadata = self.get_event_metadata(node_id, event)
        aquisition_steps = metadata.acquisition_steps if metadata else None
        if aquisition_steps:
            keys = list(aquisition_steps.keys())
            for key in keys:
                aquisition_steps_for_query = aquisition_steps[key]
                for idx in aquisition_steps_for_query:
                    if event in aquisition_steps_for_query[idx]["events_to_pull"]:
                        return key, aquisition_steps_for_query[idx]

        return None


global_event_placement_tracker: Optional[EventPlacementTracker] = None


def initialize_global_event_tracker(
    h_network_data: Optional[Dict[int, List[str]]] = None,
    h_nodes: Optional[Dict[str, List[int]]] = None,
) -> EventPlacementTracker:
    """Initialize the global event placement tracker with network data.

    Args:
        h_network_data: Dict mapping node IDs to lists of events at those nodes
                       (e.g., {6: ['A', 'E'], 7: ['B', 'C', 'D']})
        h_nodes: Dict mapping event names to lists of node IDs where they exist
                (e.g., {'A': [6, 8, 10], 'B': [7, 10]})

    Returns:
        The initialized global EventPlacementTracker instance.
    """
    global global_event_placement_tracker

    # Create new tracker instance
    global_event_placement_tracker = EventPlacementTracker()

    # Initialize from h_network_data if provided (node -> events mapping)
    if h_network_data:
        for node_id, events in h_network_data.items():
            for event in events:
                global_event_placement_tracker.add_event_at_node(
                    node_id=node_id,
                    event=event,
                    query_id=None,
                    acquisition_type="initial",
                )

    # Initialize from h_nodes if provided (event -> nodes mapping)
    elif h_nodes:
        for event, node_list in h_nodes.items():
            for node_id in node_list:
                global_event_placement_tracker.add_event_at_node(
                    node_id=node_id,
                    event=event,
                    query_id=None,
                    acquisition_type="initial",
                )

    return global_event_placement_tracker


def get_global_event_placement_tracker() -> EventPlacementTracker:
    """Get the singleton global event placement tracker instance.

    Returns:
        The global EventPlacementTracker instance.

    Raises:
        RuntimeError: If the tracker hasn't been initialized yet.
    """
    if global_event_placement_tracker is None:
        raise RuntimeError(
            "Global event placement tracker not initialized. Call initialize_global_event_tracker() first."
        )
    return global_event_placement_tracker
