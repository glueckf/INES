"""
Stack-based event tracking for placement decisions.

This module provides a lightweight alternative to the global event tracker,
using simple dict-based data structures for better performance and clarity.
"""

from typing import Dict, List, Any, Optional


def create_event_stack() -> Dict[int, Dict[str, Any]]:
    """Create an empty event stack.

    Returns:
        Empty event stack dictionary
    """
    return {}


def add_events_to_stack(
    stack: Dict[int, Dict[str, Any]],
    node_id: int,
    events: List[str],
    query_id: Any,
    acquisition_type: str,
    acquisition_steps: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    """Add events to the stack for a specific node.

    Args:
        stack: Event stack dictionary
        node_id: Node ID where events are placed
        events: List of event names
        query_id: Query/projection that acquired these events
        acquisition_type: Type of acquisition ("all_push" or "push_pull")
        acquisition_steps: Optional acquisition steps details
    """
    if node_id not in stack:
        stack[node_id] = {"events": [], "event_metadata": {}}

    # Add events to the list (avoid duplicates)
    for event in events:
        if event not in stack[node_id]["events"]:
            stack[node_id]["events"].append(event)

        # Store metadata for each event
        stack[node_id]["event_metadata"][event] = {
            "query_id": query_id,
            "acquisition_type": acquisition_type,
            "acquisition_steps": acquisition_steps,
        }


def get_events_from_stack(stack: Dict[int, Dict[str, Any]], node_id: int) -> List[str]:
    """Get all events at a specific node.

    Args:
        stack: Event stack dictionary
        node_id: Node ID to query

    Returns:
        List of event names at the node (empty list if node not found)
    """
    return stack.get(node_id, {}).get("events", [])


def get_event_metadata_from_stack(
    stack: Dict[int, Dict[str, Any]], node_id: int, event: str
) -> Optional[Dict[str, Any]]:
    """Get metadata for a specific event at a node.

    Args:
        stack: Event stack dictionary
        node_id: Node ID to query
        event: Event name to look up

    Returns:
        Event metadata dict or None if not found
    """
    return stack.get(node_id, {}).get("event_metadata", {}).get(event)


def remove_events_from_stack(
    stack: Dict[int, Dict[str, Any]], node_id: int, events: List[str]
) -> None:
    """Remove events from a specific node (useful for backtracking).

    Args:
        stack: Event stack dictionary
        node_id: Node ID to remove events from
        events: List of event names to remove
    """
    if node_id not in stack:
        return

    for event in events:
        if event in stack[node_id]["events"]:
            stack[node_id]["events"].remove(event)

        if event in stack[node_id]["event_metadata"]:
            del stack[node_id]["event_metadata"][event]

    # Clean up empty nodes
    if not stack[node_id]["events"]:
        del stack[node_id]


def clear_node_from_stack(stack: Dict[int, Dict[str, Any]], node_id: int) -> None:
    """Clear all events from a specific node.

    Args:
        stack: Event stack dictionary
        node_id: Node ID to clear
    """
    if node_id in stack:
        del stack[node_id]
