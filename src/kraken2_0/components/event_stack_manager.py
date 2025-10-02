from src.kraken2_0.state import PlacementInfo


def update_event_stack(stack: dict, node_id: int, query_id: any, placement_info: PlacementInfo):
    """vHelper function to update the stack with the eventy acquired by a new placement. """

    # The acquisition plan contains the list of events that were pulled.
    # We need to extract them.

    # TODO: Implement logic, once acquisition steps are available
    events_to_add = None

    if not events_to_add:
        return

    if node_id not in stack:
        stack[node_id] = {"events": [], "event_metadata": {}}

    for event in events_to_add:
        if event not in stack[node_id]["events"]:
            stack[node_id]["events"].append(event)

        # Store metadata about how this event became available
        stack[node_id]["event_metadata"][event] = {
            "query_id": query_id,
            "acquisition_type": placement_info.strategy
        }