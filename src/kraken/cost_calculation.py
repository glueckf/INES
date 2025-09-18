"""
Cost calculation for all-push and push-pull strategies.

This module handles the cost calculation logic for both placement
strategies, coordinating with adapters for legacy function calls.
"""

from typing import Dict, List, Any, Tuple
import re
import time
from .adapters import run_prepp
from .logging import get_kraken_logger
from .node_tracker import get_global_event_placement_tracker


# Import SimulationMode for deterministic mode checking
try:
    from INES import SimulationMode

except ImportError:
    # Fallback if import fails
    class SimulationMode:
        FULLY_DETERMINISTIC = "deterministic"
        RANDOM = "random"


logger = get_kraken_logger(__name__)


# Initialize global trackers lazily to avoid initialization issues
def _get_global_event_placement_tracker():
    """Get global event placement tracker with lazy initialization."""
    return get_global_event_placement_tracker()


def _get_global_placement_tracker():
    """Get global placement tracker with lazy initialization."""
    from .global_placement_tracker import get_global_placement_tracker

    return get_global_placement_tracker()


def calculate_final_costs_for_sending_to_sinks(
    current_push_pull_costs: float,
    current_all_push_costs: float,
    current_latency: float,
    current_transmission_ratio: float,
    placement_node: int,
    projection: Any,
    sink_nodes: List[int],
    projection_rates: Dict[Any, tuple],
    shortest_path_distances: Dict[int, Dict[int, int]],
) -> Any:
    """
    Calculate the final transmission costs for sending query results to sink nodes.

    This function takes initial push-pull costs and adds the network transmission
    costs to reach all specified sink nodes, factoring in the output rate of the
    projection and the shortest path distances.

    Args:
        current_push_pull_costs: Current cost for push-pull strategy
        current_all_push_costs: Current cost for all-push strategy
        current_latency: Current latency before adding transmission costs
        current_transmission_ratio: Current transmission efficiency ratio
        placement_node: Node ID where the projection is placed
        projection: The projection/query being processed
        sink_nodes: List of destination node IDs to send results to
        projection_rates: Dictionary mapping projections to their rate tuples
        shortest_path_distances: All-pairs shortest path distance matrix

    Returns:
        Dict: Updated results dictionary with transmission costs included
    """
    # Get the output rate for this projection

    if isinstance(projection_rates, dict) and projection in projection_rates:
        projection_output_rate = projection_rates[projection][1]
    else:
        projection_output_rate = 1  # Default fallback rate

    # Calculate distances from placement node to each sink
    distances_to_each_sink = []
    for sink_node in sink_nodes:
        distance = shortest_path_distances[placement_node][sink_node]
        distances_to_each_sink.append(distance)

    # Log transmission planning details
    logger.info(f"Projection {projection} output rate: {projection_output_rate}")
    for i, sink in enumerate(sink_nodes):
        logger.info(
            f"Distance from placement node {placement_node} to sink {sink}: {distances_to_each_sink[i]}"
        )

    # Calculate transmission costs if we have valid data
    if current_push_pull_costs is not None and distances_to_each_sink:
        # Calculate total transmission cost = sum of all distances * output rate
        total_distance_to_all_sinks = sum(distances_to_each_sink)
        total_transmission_cost = total_distance_to_all_sinks * projection_output_rate

        # Add transmission costs to both strategy costs
        final_push_pull_costs = current_push_pull_costs + total_transmission_cost
        final_all_push_costs = current_all_push_costs + total_transmission_cost

        # Calculate final latency = current latency + maximum distance to any sink
        maximum_distance_to_any_sink = max(distances_to_each_sink)
        final_latency = current_latency + maximum_distance_to_any_sink

        # Recalculate transmission efficiency ratio
        final_transmission_ratio = final_push_pull_costs / final_all_push_costs

        return (
            final_all_push_costs,
            final_push_pull_costs,
            final_latency,
            final_transmission_ratio,
        )

    else:
        # No valid costs or distances, return original results unchanged
        return (
            current_all_push_costs,
            current_push_pull_costs,
            current_latency,
            current_transmission_ratio,
        )


def calculate_prepp_with_placement(
    placement_node: int,
    projection: Any,
    query_workload,
    network,
    selectivity_rate: float,
    selectivities,
    combination_dict,
    rates,
    projection_rates,
    index_event_nodes,
    mode,
    shortest_path_distances,
    has_placed_subqueries: bool = False,
):
    input_buffer = initiate_buffer(
        placement_node,
        projection,
        network,
        selectivities,
        selectivity_rate,
        global_placement_tracker=_get_global_placement_tracker(),
        has_placed_subqueries=has_placed_subqueries,
        mycombi=combination_dict,
        rates=rates,
        projrates=projection_rates,
        index_event_nodes=index_event_nodes,
    )

    if input_buffer is None:
        logger.error("Failed to create input buffer for prePP computation")
        raise ValueError("Failed to create input buffer for prePP computation")

    # Select method and algorithm type for prePP
    method = "ppmuse"
    algorithm = "e"

    # Check if we should use deterministic behavior
    config_mode = mode.value
    is_deterministic = config_mode == SimulationMode.FULLY_DETERMINISTIC

    # Track prepp execution time
    from .state import get_kraken_timing_tracker
    timing_tracker = get_kraken_timing_tracker()
    prepp_start_time = time.time()

    results = run_prepp(
        input_buffer=input_buffer,
        method=method,
        algorithm=algorithm,
        samples=0,
        top_k=0,
        runs=1,
        plan_print=True,
        all_pairs=shortest_path_distances,
        is_deterministic=is_deterministic,
    )

    prepp_end_time = time.time()
    prepp_duration = prepp_end_time - prepp_start_time
    timing_tracker.add_prepp_time(prepp_duration)

    logger.debug(f"PrePP computation took {prepp_duration:.3f}s for node {placement_node}")

    # process results
    if results and len(results) >= 5:
        return process_results_from_prepp(
            results, query=projection, node=placement_node, workload=query_workload
        )
    else:
        logger.warning("No valid prePP results returned, using fallback")
        raise ValueError("Invalid prePP results")


def initiate_buffer(
    node,
    projection,
    network,
    selectivities,
    selection_rate,
    global_placement_tracker=None,
    has_placed_subqueries: bool = False,
    mycombi=None,
    rates=None,
    projrates=None,
    index_event_nodes=None,
) -> Any:
    """
    Generate a configuration buffer similar to generate_eval_plan but for a single projection.
    Creates evaluation dictionaries, combination mappings, and configuration data.

    Args:
        node: The target node ID for placement
        projection: The projection/query being processed
        network: List of network nodes
        selectivities: Dictionary of selectivity values
        selection_rate: Calculated selection rate for the projection

    Returns:
        io.StringIO: Configuration buffer containing the generated plan
    """
    import io

    # Safety checks
    if not _validate_buffer_inputs(
        node, projection, network, selectivities, selection_rate
    ):
        return None

    try:
        # Initialize dictionaries similar to generate_eval_plan
        evaluation_dict = _initialize_evaluation_dict(network)
        combination_dict = {}
        forwarding_dict = {}
        selection_rate_dict = {}
        filter_dict = {}
        sink_dict = {}

        # Process the projection (mimicking the loop over myplan.projections)
        if projection:
            projection_str = str(projection)

            # Extract filters if available
            filters = _extract_projection_filters(projection)
            if filters:
                filter_dict.update(filters)

            # Set evaluation node for this projection
            evaluation_dict[node].append(projection_str)

            # Create combination dictionary mapping
            combination_keys = _extract_combination_keys(projection, mycombi)
            combination_dict[projection_str] = combination_keys

            # Store selection rate
            selection_rate_dict[projection_str] = selection_rate

            # Create sink dictionary
            sinks = _extract_projection_sinks(projection, node)
            sink_dict[projection_str] = [sinks, ""]

            # Process forwarding dictionary (simplified version)
            forwarding_dict = _process_forwarding_dict(projection, forwarding_dict)

        # Generate the configuration plan
        workload = [projection_str] if projection else []

        # Create configuration buffer
        config_buffer = io.StringIO()

        # Generate plan content (placeholder for actual generatePlan function)
        plan_content = _generate_plan_content(
            network=network,
            selectivities=selectivities,
            workload=workload,
            combination_dict=combination_dict,
            sink_dict=sink_dict,
            selection_rate_dict=selection_rate_dict,
            projection=projection,
            has_placed_subqueries=has_placed_subqueries,
            mycombi=mycombi,
            projrates=projrates,
        )

        config_buffer.write(plan_content)
        config_buffer.seek(0)

        return config_buffer

    except Exception as e:
        logger.error(f"Error creating buffer: {str(e)}")
        return None


def _validate_buffer_inputs(node, projection, network, selectivities, selection_rate):
    """Validate all input parameters for initiate_buffer."""
    if not isinstance(network, list) or not network:
        logger.warning(f"Invalid network: expected non-empty list, got {type(network)}")
        return False

    if not isinstance(node, int) or node < 0 or node >= len(network):
        logger.warning(f"Invalid node ID: {node}, network has {len(network)} nodes")
        return False

    if not isinstance(selectivities, dict):
        logger.warning(
            f"Invalid selectivities: expected dict, got {type(selectivities)}"
        )
        return False

    if not isinstance(selection_rate, (int, float)) or selection_rate < 0:
        logger.warning(f"Invalid selection rate: {selection_rate}")
        return False

    return True


def _initialize_evaluation_dict(network):
    """Initialize evaluation dictionary with empty lists for each network node."""
    return {i: [] for i in range(len(network))}


def _extract_projection_filters(projection):
    """Extract filters from projection if available."""
    flt = getattr(projection, "Filters", None)
    if not flt:
        return {}
    if isinstance(flt, dict):
        # Return as-is (or dict(flt) if you need a copy)
        return flt
    try:
        # Fast path in C if every item is a pair (k, v)
        return dict(flt)
    except (TypeError, ValueError):
        # Fall back to safe parsing below
        pass

    # Robust fallback (handles items longer than 2, etc.)
    out = {}
    for t in flt:
        if len(t) >= 2:
            out[t[0]] = t[1]
    return out


_sentinel = object()


def _extract_combination_keys(projection, mycombi):
    """Extract combination keys from projection."""
    # 1) Single lookup in mycombi (handles falsy values correctly via sentinel)
    val = mycombi.get(projection, _sentinel)
    if val is not _sentinel:
        return list(val)  # ensure list, like original

    # 2) Try combination.keys() if present
    comb = getattr(projection, "combination", None)
    if comb is not None:
        keys = getattr(comb, "keys", None)
        if keys is not None:
            return [str(k) for k in keys()]

    # 3) Fallback to children if present
    children = getattr(projection, "children", None)
    if children is not None:
        return [str(c) for c in children]

    # 4) Last resort: the projection itself
    return [str(projection)]


def _extract_projection_sinks(projection, default_node):
    """Extract sink nodes from projection or use default."""
    if hasattr(projection, "sinks") and projection.sinks:
        return projection.sinks
    else:
        return [default_node]


def _process_forwarding_dict(projection, forwarding_dict):
    """Process forwarding dictionary for the projection (simplified)."""
    # Simplified version - would need actual instance processing logic
    if hasattr(projection, "combination"):
        # This would contain the actual forwarding logic from the original
        pass
    return forwarding_dict


def _generate_plan_content(
    network,
    selectivities,
    workload,
    combination_dict,
    sink_dict,
    selection_rate_dict,
    projection=None,
    has_placed_subqueries=False,
    mycombi=None,
    projrates=None,
):
    """Generate the actual plan content string."""
    lines = []

    # Convert Node objects to node IDs
    def extract_node_ids(node_list):
        if node_list is None:
            return None
        if not hasattr(node_list, "__iter__") or isinstance(node_list, str):
            return node_list

        ids = []
        for item in node_list:
            if hasattr(item, "id"):
                ids.append(item.id)
            elif hasattr(item, "nodeID"):
                ids.append(item.nodeID)
            elif hasattr(item, "node_id"):
                ids.append(item.node_id)
            elif isinstance(item, int):
                ids.append(item)
            else:
                # Try to get index from network list
                try:
                    if item in network:
                        ids.append(network.index(item))
                    else:
                        ids.append(str(item))
                except ValueError:
                    ids.append(str(item))
        return ids if ids else None

    # First pass: collect all children relationships to build parent map
    parent_map = {}  # child_id -> [parent_ids]
    all_children = {}  # node_id -> [child_ids]

    for i, node in enumerate(network):
        children_raw = (
            getattr(node, "children", None)
            or getattr(node, "child", None)
            or getattr(node, "Child", None)
            or getattr(node, "childs", None)
        )

        children = extract_node_ids(children_raw)
        all_children[i] = children

        if children:
            for child_id in children:
                if child_id not in parent_map:
                    parent_map[child_id] = []
                parent_map[child_id].append(i)

    # Add network information
    lines.append("network")
    for i, node in enumerate(network):
        lines.append(f"Node {i} Node {i}")

        # Try different possible attribute names for computational power
        comp_power = (
            getattr(node, "computational_power", None)
            or getattr(node, "computationalPower", None)
            or getattr(node, "comp_power", None)
            or "inf"
        )

        # Try different possible attribute names for memory
        memory = getattr(node, "memory", None) or getattr(node, "Memory", None) or "inf"

        # Try different possible attribute names for event rates
        event_rates = (
            getattr(node, "event_rates", None)
            or getattr(node, "eventrates", None)
            or getattr(node, "Eventrates", None)
            or getattr(node, "rates", None)
            or [0] * 6
        )

        # Get parents from the parent map we built
        parents = parent_map.get(i, None)

        # Get children
        children = all_children[i]

        # Try different possible attribute names for siblings
        siblings_raw = (
            getattr(node, "siblings", None)
            or getattr(node, "Siblings", None)
            or getattr(node, "sibling", None)
        )
        siblings = extract_node_ids(siblings_raw)

        lines.append(f"Computational Power: {comp_power}")
        lines.append(f"Memory: {memory}")
        lines.append(
            f"Eventrates: {list(event_rates) if hasattr(event_rates, '__iter__') and not isinstance(event_rates, str) else event_rates}"
        )
        lines.append(f"Parents: {parents}")
        lines.append(f"Child: {children}")
        lines.append(f"Siblings: {siblings}")
        lines.append("")

    # Add selectivities
    lines.append("selectivities")
    lines.append(str(selectivities))
    lines.append("")

    # Add queries
    lines.append("queries")
    for query in workload:
        lines.append(str(query))
    lines.append("")

    # Add muse graph
    if workload and selection_rate_dict:
        lines.append("muse graph")

        # Check if we have placed subqueries and need to handle them differently
        if has_placed_subqueries and projection and mycombi and projrates:
            global_placement_tracker = _get_global_placement_tracker()
            # Get subqueries for this projection
            if projection in mycombi:
                subqueries = mycombi[projection]

                # Add placed subqueries first
                for subquery in subqueries:
                    if hasattr(
                        subquery, "leafs"
                    ) and global_placement_tracker.has_placement_for(subquery):
                        placement_decision = (
                            global_placement_tracker.get_best_placement(subquery)
                        )
                        subquery_rate = projrates.get(subquery, 0.001)
                        if len(subquery_rate) > 1:
                            subquery_rate = subquery_rate[0]

                        # Get primitive events for subquery
                        if hasattr(subquery, "leafs"):
                            primitive_events = subquery.leafs()
                        else:
                            primitive_events = [str(subquery)]

                        combination_str = "; ".join(primitive_events)
                        lines.append(
                            f"SELECT {subquery} FROM {combination_str} ON {{{placement_decision.node}}} WITH selectionRate= {subquery_rate}"
                        )

                # Add main query with virtual events (replace placed subqueries with their virtual names)
                main_query = workload[0]
                main_rate = selection_rate_dict.get(main_query, 0)

                # Create combination for main query
                main_combination = []
                for elem in subqueries:
                    if hasattr(
                        elem, "leafs"
                    ) and global_placement_tracker.has_placement_for(elem):
                        # This subquery is placed, so we reference it as virtual event
                        main_combination.append(str(elem))
                    else:
                        # This is a primitive event
                        main_combination.append(str(elem))

                combination_str = "; ".join(main_combination)
                lines.append(
                    f"SELECT {main_query} FROM {combination_str} ON {{0}} WITH selectionRate= {main_rate}"
                )
            else:
                # Fallback to normal handling
                _add_normal_muse_graph_entry(
                    lines, workload, selection_rate_dict, sink_dict, combination_dict
                )
        else:
            # Normal case without subqueries
            _add_normal_muse_graph_entry(
                lines, workload, selection_rate_dict, sink_dict, combination_dict
            )

    return "\n".join(lines)


def _add_normal_muse_graph_entry(
    lines, workload, selection_rate_dict, sink_dict, combination_dict
):
    """Add normal muse graph entry without subqueries."""
    query = workload[0]
    rate = selection_rate_dict.get(query, 0)
    sink_info = sink_dict.get(query, [[], ""])
    sink_nodes = sink_info[0] if sink_info[0] else [0]

    # Get combination for query - extract primitive events
    if hasattr(query, "leafs"):
        primitive_events = query.leafs()
    else:
        primitive_events = combination_dict.get(query, [query])

    combination_str = "; ".join(str(e) for e in primitive_events)

    lines.append(
        f"SELECT {query} FROM {combination_str} ON {{{', '.join(map(str, sink_nodes))}}} WITH selectionRate= {rate}"
    )


def _expand_to_primitives(element, combination_dict):
    """
    Recursively expand a subquery element to its primitive events.

    Args:
        element: The element to expand (could be a subquery)
        combination_dict: Dictionary mapping projections to their combinations

    Returns:
        List of primitive event strings
    """
    if element not in combination_dict:
        # This is already a primitive event
        return [str(element)]

    primitives = []
    for sub_elem in combination_dict[element]:
        if sub_elem in combination_dict:
            # Recursively expand this sub-element
            primitives.extend(_expand_to_primitives(sub_elem, combination_dict))
        else:
            # This is a primitive event
            primitives.append(str(sub_elem))

    return primitives


def process_results_from_prepp(results, query, node, workload):
    qkey = str(query)

    # EAFP indexing with cheap fallbacks
    try:
        all_push_costs = results[4]
    except IndexError:
        all_push_costs = 0

    try:
        computing_time = results[1]
    except IndexError:
        computing_time = 0

    try:
        steps_by_proj = results[6]
    except IndexError:
        steps_by_proj = None

    if not steps_by_proj or qkey not in steps_by_proj:
        raise ValueError("No acquisition steps found in prePP results")

    steps = steps_by_proj[qkey]  # expected: dict[str, dict]
    vals = steps.values()  # iterate values directly

    # Single pass accumulation (fewer Python-level operations)
    total_plan_costs = 0
    total_plan_latency = 0
    for s in vals:
        # dict.get avoids KeyError and stays fast
        total_plan_costs += s.get("total_step_costs", 0)
        total_plan_latency += s.get("total_latency", 0)

    transmission_ratio = (total_plan_costs / all_push_costs) if all_push_costs else 0.0

    return {
        "node_for_placement": node,
        "projection": query,
        "all_push_costs": all_push_costs,
        "push_pull_costs": total_plan_costs,
        "latency": total_plan_latency,
        "computing_time": computing_time,
        "transmission_ratio": transmission_ratio,
        "aquisition_steps": steps,  # same key as before
    }


def calculate_costs(
    placement_node: int,
    projection: Any,
    query_workload,
    network,
    selectivities,
    combination_dict,
    rates,
    projection_rates,
    index_event_nodes,
    mode,
    shortest_path_distances,
    sink_nodes,
    has_placed_subqueries: bool = False,
) -> Any:
    """
    Calculate all placement costs for a given node and projection.

    This function orchestrates the cost calculation process by:
    1. Running initial cost calculation via prePP
    2. Adjusting costs for locally available events
    3. Adding transmission costs to sinks if needed

    Args:
        placement_node: Node ID where projection will be placed
        projection: The projection/query being processed
        query_workload: List of queries in the workload
        network: List of network nodes
        selectivities: Dictionary of selectivity values
        combination_dict: Dictionary mapping projections to combinations
        rates: Event rate dictionary
        projection_rates: Projection output rates
        index_event_nodes: Event node index mapping
        mode: Simulation mode (deterministic/random)
        shortest_path_distances: All-pairs shortest path distances
        sink_nodes: List of sink node IDs
        has_placed_subqueries: Whether subqueries are already placed

    Returns:
        Tuple: (all_push_costs, push_pull_costs, latency, computing_time,
                transmission_ratio, acquisition_steps)
    """
    logger.debug(
        f"Calculating costs for node {placement_node}, projection {projection}"
    )

    selectivity_rate = projection_rates.get(projection, (0.0, 0.0))[0]

    # Initial cost calculation
    results = calculate_prepp_with_placement(
        placement_node=placement_node,
        projection=projection,
        query_workload=query_workload,
        network=network,
        selectivity_rate=selectivity_rate,
        selectivities=selectivities,
        combination_dict=combination_dict,
        rates=rates,
        projection_rates=projection_rates,
        index_event_nodes=index_event_nodes,
        mode=mode,
        shortest_path_distances=shortest_path_distances,
        has_placed_subqueries=has_placed_subqueries,
    )

    # Adjust costs if some events are already available at the node
    all_push_costs, push_pull_costs, latency, transmission_ratio = (
        handle_locally_available_events(
            results=results, placement_node=placement_node, projection=projection
        )
    )

    # Check if we need to send results to sinks (e.g., cloud)
    needs_to_be_sent_to_cloud = check_if_projection_needs_to_be_sent_to_cloud(
        projection=projection,
        query_workload=query_workload,
        placement_node=placement_node,
        sink_nodes=sink_nodes,
    )

    if needs_to_be_sent_to_cloud:
        # Add transmission costs to sinks
        all_push_costs, push_pull_costs, latency, transmission_ratio = (
            calculate_final_costs_for_sending_to_sinks(
                current_push_pull_costs=push_pull_costs,
                current_all_push_costs=all_push_costs,
                current_latency=latency,
                current_transmission_ratio=transmission_ratio,
                placement_node=placement_node,
                projection=projection,
                sink_nodes=sink_nodes,
                projection_rates=projection_rates,
                shortest_path_distances=shortest_path_distances,
            )
        )

    # Return final costs and metrics (no modifications for global optimization)
    return (
        all_push_costs,
        push_pull_costs,
        latency,
        results["computing_time"],
        transmission_ratio,
        results["aquisition_steps"],
    )


def check_if_projection_needs_to_be_sent_to_cloud(
    projection, query_workload, placement_node, sink_nodes
) -> bool:
    """
    Check if projection results need to be sent to cloud/sink nodes.

    Args:
        projection: The projection being evaluated
        query_workload: List of queries in the workload
        placement_node: Node where projection is placed
        sink_nodes: List of sink node IDs

    Returns:
        bool: True if projection needs to be sent to sinks
    """
    # If the projection is part of the workload and not placed at a sink, it needs to be sent to the cloud
    return projection in query_workload and placement_node not in sink_nodes


def handle_locally_available_events(
    results, placement_node, projection
) -> Tuple[float, float, float, float]:
    """
    Handle cost adjustments for locally available events at placement node.

    Args:
        results: Results from initial cost calculation
        placement_node: Node ID where projection is placed
        projection: The projection being processed

    Returns:
        Tuple: (adjusted_all_push_costs, adjusted_push_pull_costs, latency, transmission_ratio)
    """
    available_events = set(
        _get_global_event_placement_tracker().get_events_at_node(placement_node)
    )
    needed_events = get_events_for_projection(projection)

    if available_events & needed_events:  # If there's intersection
        return handle_intersection(results, available_events, needed_events, logger)
    else:
        return (
            results["all_push_costs"],
            results["push_pull_costs"],
            results["latency"],
            results["transmission_ratio"],
        )


def handle_intersection(
    results, available_events, needed_events, logger
) -> Tuple[float, float, float, float]:
    """
    Handle cost adjustments when some events are already available at the placement node.

    This function removes acquisition costs for events that don't need to be acquired
    because they're already present at the node. It handles both primitive events (A, B, C)
    and complex events (SEQ(A, B), AND(C, D)) by extracting the primitive components.

    Args:
        results: Cost calculation results dictionary
        available_events: Set of events available at the node
        needed_events: Set of events needed for the projection
        logger: Logger instance

    Returns:
        Tuple: (adjusted_all_push_costs, adjusted_push_pull_costs, latency, transmission_ratio)
    """
    # Extract current costs from results
    current_all_push_costs = results["all_push_costs"]
    current_push_pull_costs = results["push_pull_costs"]
    current_latency = results["latency"]
    current_transmission_ratio = results["transmission_ratio"]
    current_acquisition_steps = results["aquisition_steps"]

    # Find events that are both available and needed (intersection)
    events_already_available = available_events & needed_events

    # Track total cost adjustments
    total_cost_adjustment = 0.0

    # Go through each acquisition step and check if we can skip it
    for step_index, step_details in current_acquisition_steps.items():
        events_to_pull_in_this_step = step_details.get("events_to_pull", [])
        step_total_cost = step_details.get("total_step_costs", 0.0)

        # Extract primitive events from each event in this step
        all_primitive_events_in_step = set()
        for event_or_subquery in events_to_pull_in_this_step:
            if isinstance(event_or_subquery, str):
                # Handle string representations like 'SEQ(A, B)' or simple events like 'A'
                primitive_events = determine_all_primitive_events_of_projection(
                    event_or_subquery
                )
                all_primitive_events_in_step.update(primitive_events)
            else:
                # Handle object representations - try to get string representation
                primitive_events = determine_all_primitive_events_of_projection(
                    str(event_or_subquery)
                )
                all_primitive_events_in_step.update(primitive_events)

        # Check if any of the primitive events in this step are already available
        events_we_can_skip = all_primitive_events_in_step & events_already_available

        if events_we_can_skip:
            # If all primitive events in this step are already available, skip entire step cost
            if (
                all_primitive_events_in_step <= events_already_available
            ):  # All events in step are available
                total_cost_adjustment += step_total_cost
            else:
                # Partial adjustment - some primitive events in step are available
                fraction_available = len(events_we_can_skip) / len(
                    all_primitive_events_in_step
                )
                partial_adjustment = step_total_cost * fraction_available
                total_cost_adjustment += partial_adjustment

    # Apply cost adjustments
    adjusted_push_pull_costs = current_push_pull_costs - total_cost_adjustment

    # TODO: Fix all-push cost adjustment logic because it should not be reduced only by push pull change
    adjusted_all_push_costs = current_all_push_costs - total_cost_adjustment

    # Ensure costs don't go negative
    adjusted_push_pull_costs = max(0.0, adjusted_push_pull_costs)
    adjusted_all_push_costs = max(0.0, adjusted_all_push_costs)

    # Recalculate transmission ratio if needed
    if adjusted_all_push_costs > 0:
        adjusted_transmission_ratio = adjusted_push_pull_costs / adjusted_all_push_costs
    else:
        adjusted_transmission_ratio = current_transmission_ratio

    return (
        adjusted_all_push_costs,
        adjusted_push_pull_costs,
        current_latency,
        adjusted_transmission_ratio,
    )


def determine_all_primitive_events_of_projection(projection) -> List[str]:
    """
    Extract primitive events from a projection string like 'SEQ(A, B)' -> ['A', 'B'].

    Args:
        projection: Projection string to parse

    Returns:
        List of primitive event strings
    """
    given_predicates = str(projection).replace("AND", "")
    given_predicates = given_predicates.replace("SEQ", "")
    given_predicates = given_predicates.replace("(", "")
    given_predicates = given_predicates.replace(")", "")
    given_predicates = re.sub(r"[0-9]+", "", given_predicates)
    given_predicates = given_predicates.replace(" ", "")
    if "," in given_predicates:
        return given_predicates.split(",")
    else:
        # Handle single events without comma
        return list(given_predicates)


# Do the import once at module load (cheaper than inside the function each call)
try:
    from helper.Tree import PrimEvent as _PrimEvent
except ImportError:
    _PrimEvent = None


def get_events_for_projection(projection):
    """
    Iterative DFS; returns set[str] of primitive event types.
    Uses leafs() fast-path if available.
    """
    # Fast path: many of your nodes seem to implement .leafs()
    leafs = getattr(projection, "leafs", None)
    if callable(leafs):
        # Assumes leafs() returns iterable of primitive event names
        return set(leafs())

    result = set()
    add = result.add  # bind once
    stack = [projection]

    PrimEvent = _PrimEvent  # local binding is faster in tight loops
    if PrimEvent is None:
        # Fallback if import failed
        PrimEvent = type(None)  # dummy type that won't match anything

    while stack:
        node = stack.pop()

        # Primitive leaf
        if isinstance(node, PrimEvent):
            add(node.evtype)
            continue

        # Composite: push children (if any)
        children = getattr(node, "children", None)
        if not children:
            # If some implementations store evtype on non-PrimEvent leaves
            ev = getattr(node, "evtype", None)
            if ev is not None:
                add(ev)
            continue

        # Add children; collect primitive ones immediately
        for child in children:
            if isinstance(child, PrimEvent):
                add(child.evtype)
            else:
                stack.append(child)

    return result
