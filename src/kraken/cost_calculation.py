"""
Cost calculation for all-push and push-pull strategies.

This module handles the cost calculation logic for both placement
strategies, coordinating with adapters for legacy function calls.
"""

from typing import Dict, List, Any, Tuple, Set, Optional
import re
import time
import hashlib
from .adapters import run_prepp
from .event_stack import get_events_from_stack, get_event_metadata_from_stack
from .logging import get_kraken_logger


# Import SimulationMode for deterministic mode checking
try:
    from INES import SimulationMode

except ImportError:
    # Fallback if import fails
    class SimulationMode:
        FULLY_DETERMINISTIC = "deterministic"
        RANDOM = "random"


logger = get_kraken_logger(__name__)

# PrePP result cache for identical computations
_prepp_cache: Dict[str, Any] = {}
_cache_hits = 0
_cache_misses = 0


def _create_prepp_cache_key(
    placement_node: int,
    projection: Any,
    selectivity_rate: float,
    selectivities: Dict,
    is_deterministic: bool,
) -> str:
    """Create a deterministic cache key for prepp computation."""
    # Use stable string representation for projection
    proj_str = str(projection)

    # Create deterministic representation of selectivities
    sel_items = sorted(selectivities.items()) if selectivities else []
    sel_str = str(sel_items)

    # Combine all deterministic inputs
    key_data = (
        f"{placement_node}|{proj_str}|{selectivity_rate}|{sel_str}|{is_deterministic}"
    )

    # Use hash to keep key size manageable
    return hashlib.md5(key_data.encode()).hexdigest()


def _get_prepp_cache_stats() -> Dict[str, int]:
    """Get cache performance statistics."""
    total = _cache_hits + _cache_misses
    hit_rate = (_cache_hits / total * 100) if total > 0 else 0
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "total": total,
        "hit_rate_percent": hit_rate,
        "cache_size": len(_prepp_cache),
    }


# Removed global tracker - now using stack-based approach


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
    nodes_per_primitive_event,
    has_placed_subqueries: bool = False,
    placed_subqueries: Optional[Dict[Any, int]] = None,
    local_rate_lookup: Optional[Dict[str, Dict[int, float]]] = None,
):
    # Check if we should use deterministic behavior
    config_mode = mode.value
    is_deterministic = config_mode == SimulationMode.FULLY_DETERMINISTIC

    # Try cache first for deterministic computations
    cache_key = None
    global _cache_hits, _cache_misses

    if is_deterministic:
        cache_key = _create_prepp_cache_key(
            placement_node,
            projection,
            selectivity_rate,
            selectivities,
            is_deterministic,
        )

        if cache_key in _prepp_cache:
            _cache_hits += 1
            logger.debug(
                f"PrePP cache hit for node {placement_node} (key: {cache_key[:8]}...)"
            )
            return _prepp_cache[cache_key]
        else:
            _cache_misses += 1

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
        placed_subqueries=placed_subqueries,
        local_rate_lookup=local_rate_lookup,
    )

    if input_buffer is None:
        logger.error("Failed to create input buffer for prePP computation")
        raise ValueError("Failed to create input buffer for prePP computation")

    # Select method and algorithm type for prePP
    method = "ppmuse"
    algorithm = "e"

    # Track prepp execution time
    from .state import get_kraken_timing_tracker

    timing_tracker = get_kraken_timing_tracker()
    prepp_start_time = time.time()

    all_push_costs, all_push_latency = calculate_central_costs_for_placement(
        placement_node,
        projection,
        network,
        selectivities,
        selectivity_rate,
        has_placed_subqueries=has_placed_subqueries,
        mycombi=combination_dict,
        rates=rates,
        projrates=projection_rates,
        index_event_nodes=index_event_nodes,
        placed_subqueries=placed_subqueries,
        shortest_path_distances=shortest_path_distances,
        nodes_per_primitive_event=nodes_per_primitive_event,
        local_rate_lookup=local_rate_lookup,
    )

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
        projection=projection,
        combination_dict=combination_dict,
    )

    # Update prepp results for all push with the correct results from central cost calculation
    if results and len(results) >= 5:
        results[4] = all_push_costs
        results[5] = all_push_latency

    prepp_end_time = time.time()
    prepp_duration = prepp_end_time - prepp_start_time
    timing_tracker.add_prepp_time(prepp_duration)

    logger.debug(
        f"PrePP computation took {prepp_duration:.3f}s for node {placement_node}"
    )

    # process results
    if results and len(results) >= 5:
        processed_results = process_results_from_prepp(
            results, query=projection, node=placement_node, workload=query_workload
        )

        # Cache results for future use (only for deterministic mode)
        if is_deterministic and cache_key is not None:
            _prepp_cache[cache_key] = processed_results
            logger.debug(
                f"Cached PrePP result for node {placement_node} (key: {cache_key[:8]}...)"
            )

        return processed_results
    else:
        logger.warning("No valid prePP results returned, using fallback")
        raise ValueError("Invalid prePP results")


def calculate_central_costs_for_placement(
    placement_node: int,
    projection: Any,
    network,
    selectivities: Dict,
    selection_rate: float,
    has_placed_subqueries: bool = False,
    mycombi=None,
    rates=None,
    projrates=None,
    index_event_nodes=None,
    nodes_per_primitive_event=None,
    placed_subqueries: Optional[Dict[Any, int]] = None,
    shortest_path_distances: Dict[int, Dict[int, int]] = None,
    local_rate_lookup: Optional[Dict[str, Dict[int, float]]] = None,
) -> Tuple[float, float]:
    dependencies = mycombi.get(projection, []) if mycombi else []

    # Optimized latency calculation using the new data structure
    latency = 0.0
    total_cost = 0.0

    for dependency in dependencies:
        if local_rate_lookup and dependency in local_rate_lookup:
            for source_node in local_rate_lookup[dependency]:
                # O(1) lookup of local rate for this dependency at this source node
                rate = local_rate_lookup[dependency][source_node]

                # O(1) lookup of shortest path distance
                hops = shortest_path_distances[source_node][placement_node]

                # Calculate cost contribution
                cost = hops * rate
                total_cost += cost

                # Update maximum latency (latency is determined by the maximum hops)
                latency = max(hops, latency)
        elif dependency in projrates and dependency in placed_subqueries:
            hops = shortest_path_distances[placed_subqueries[dependency]][
                placement_node
            ]
            rate = projrates[dependency][1]

            latency = max(hops, latency)
            cost = hops * rate
            total_cost += cost

    return total_cost, latency


def calculate_push_acquisition_steps(
    placement_node: int,
    projection: Any,
    mycombi=None,
    rates=None,
    projrates=None,
    placed_subqueries: Optional[Dict[Any, int]] = None,
    shortest_path_distances: Dict[int, Dict[int, int]] = None,
    local_rate_lookup: Optional[Dict[str, Dict[int, float]]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Calculate acquisition steps for all-push strategy.

    Args:
        placement_node: Target placement node
        projection: Projection being placed
        mycombi: Dependency mappings
        rates: Event rates
        projrates: Projection rates
        placed_subqueries: Already placed subqueries
        shortest_path_distances: Distance matrix
        local_rate_lookup: Local rate lookup structure

    Returns:
        Dict mapping step number to acquisition step details
    """
    dependencies = mycombi.get(projection, []) if mycombi else []
    acquisition_steps = {0: _create_empty_step()}

    step = acquisition_steps[0]
    events_to_pull = []
    pull_response_details = {}
    total_cost = 0.0
    max_latency = 0.0

    for dependency in dependencies:
        if local_rate_lookup and dependency in local_rate_lookup:
            events_to_pull.append(str(dependency))
            event_sources = []
            event_cost = 0.0
            event_latency = 0

            for source_node in local_rate_lookup[dependency]:
                rate = local_rate_lookup[dependency][source_node]
                hops = shortest_path_distances[source_node][placement_node]
                raw_cost = hops * rate

                event_sources.append(
                    {
                        "source_node": source_node,
                        "distance": hops,
                        "base_rate": rate,
                        "raw_cost": raw_cost,
                    }
                )

                event_cost += raw_cost
                event_latency = max(event_latency, hops)

            pull_response_details[str(dependency)] = {
                "raw_cost": event_cost,
                "cost_with_selectivity": event_cost,
                "selectivity_applied": 1.0,
                "latency": event_latency,
                "sources": event_sources,
            }

            total_cost += event_cost
            max_latency = max(max_latency, event_latency)

        elif dependency in projrates and dependency in placed_subqueries:
            events_to_pull.append(str(dependency))
            source_node = placed_subqueries[dependency]
            rate = projrates[dependency][1]
            hops = shortest_path_distances[source_node][placement_node]
            raw_cost = hops * rate

            pull_response_details[str(dependency)] = {
                "raw_cost": raw_cost,
                "cost_with_selectivity": raw_cost,
                "selectivity_applied": 1.0,
                "latency": hops,
                "sources": [
                    {
                        "source_node": source_node,
                        "distance": hops,
                        "base_rate": rate,
                        "raw_cost": raw_cost,
                    }
                ],
            }

            total_cost += raw_cost
            max_latency = max(max_latency, hops)

    step["events_to_pull"] = events_to_pull
    step["detailed_cost_contribution"]["pull_response"] = pull_response_details
    step["pull_response_costs"] = total_cost
    step["pull_response_latency"] = max_latency
    step["total_step_costs"] = total_cost
    step["total_latency"] = float(max_latency)

    return acquisition_steps


def _create_empty_step() -> Dict[str, Any]:
    """Create an empty acquisition step structure."""
    return {
        "pull_set": [],
        "events_to_pull": [],
        "pull_request_costs": 0.0,
        "pull_request_latency": 0.0,
        "pull_response_costs": 0.0,
        "pull_response_latency": 0,
        "total_step_costs": 0.0,
        "total_latency": 0.0,
        "already_at_node": None,
        "acquired_by_query": None,
        "detailed_cost_contribution": {
            "pull_request": {},
            "pull_response": {},
        },
    }


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
    placed_subqueries: Optional[Dict[Any, int]] = None,
    local_rate_lookup: Optional[Dict[str, Dict[int, float]]] = None,
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
        global_placement_tracker: Legacy tracker (deprecated, use placed_subqueries)
        has_placed_subqueries: Whether subqueries have been placed
        mycombi: Subquery combination mappings
        rates: Rate mappings
        projrates: Projection rate mappings
        index_event_nodes: Event node indices
        placed_subqueries: Dict mapping subqueries to their placement node IDs

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
            global_placement_tracker=global_placement_tracker,
            placed_subqueries=placed_subqueries,
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
    global_placement_tracker=None,
    placed_subqueries: Optional[Dict[Any, int]] = None,
):
    """Generate the actual plan content string."""
    lines = []

    # Create O(1) lookup dictionary for network nodes to avoid O(n) index() calls
    network_lookup = {}
    for i, node in enumerate(network):
        network_lookup[node] = i

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
                # Use O(1) lookup instead of O(n) network.index()
                if item in network_lookup:
                    ids.append(network_lookup[item])
                else:
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
            # Determine the source of placement information
            if placed_subqueries is not None:
                # Use the new parameter (preferred)
                placement_source = placed_subqueries
            elif global_placement_tracker is not None:
                # Fallback to legacy tracker for backward compatibility
                placement_source = global_placement_tracker
            else:
                # No placement information available, fallback to normal handling
                placement_source = None

            if placement_source is not None:
                # Get subqueries for this projection
                if projection in mycombi:
                    subqueries = mycombi[projection]

                    # Add placed subqueries first
                    for subquery in subqueries:
                        placement_node = None

                        # Check placement based on source type
                        if (
                            isinstance(placement_source, dict)
                            and hasattr(subquery, "leafs")
                            and subquery in placement_source
                        ):
                            # New parameter: direct dictionary lookup
                            placement_node = placement_source[subquery]
                        elif (
                            hasattr(placement_source, "has_placement_for")
                            and hasattr(subquery, "leafs")
                            and placement_source.has_placement_for(subquery)
                        ):
                            # Legacy tracker: use tracker methods
                            placement_decision = placement_source.get_best_placement(
                                subquery
                            )
                            placement_node = placement_decision.node

                        if placement_node is not None:
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
                                f"SELECT {subquery} FROM {combination_str} ON {{{placement_node}}} WITH selectionRate= {subquery_rate}"
                            )

                    # Add main query with virtual events
                    main_query = workload[0]
                    main_rate = selection_rate_dict.get(main_query, 0)

                    # Create combination for main query
                    main_combination = []
                    for elem in subqueries:
                        is_placed = False

                        # Check placement based on source type
                        if (
                            isinstance(placement_source, dict)
                            and hasattr(elem, "leafs")
                            and elem in placement_source
                        ):
                            # New parameter: direct dictionary check
                            is_placed = True
                        elif (
                            hasattr(placement_source, "has_placement_for")
                            and hasattr(elem, "leafs")
                            and placement_source.has_placement_for(elem)
                        ):
                            # Legacy tracker: use tracker methods
                            is_placed = True

                        if is_placed:
                            # This subquery is placed, reference as virtual event
                            main_combination.append(str(elem))
                        else:
                            # This is a primitive event
                            main_combination.append(str(elem))

                    combination_str = "; ".join(main_combination)
                    lines.append(
                        f"SELECT {main_query} FROM {combination_str} ON {{0}} WITH selectionRate= {main_rate}"
                    )
                else:
                    # Fallback to normal handling if projection not in mycombi
                    _add_normal_muse_graph_entry(
                        lines,
                        workload,
                        selection_rate_dict,
                        sink_dict,
                        combination_dict,
                    )
            else:
                # Fallback to normal handling if no placement source
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

    try:
        all_push_latency = results[5]
    except IndexError:
        all_push_latency = 0

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
        "all_push_latency": all_push_latency,
        "push_pull_latency": total_plan_latency,
        "computing_time": computing_time,
        "transmission_ratio": transmission_ratio,
        "aquisition_steps": steps,  # same key as before
    }


def calculate_costs(
    placement_node: int,
    current_projection: Any,
    query_workload,
    network_data_nodes,
    pairwise_selectivities,
    dependencies_per_projection,
    global_event_rates,
    projection_rates_selectivity,
    index_event_nodes,
    simulation_mode,
    pairwise_distance_matrix,
    sink_nodes,
    nodes_per_primitive_event,
    has_placed_subqueries: bool = False,
    placed_subqueries: Optional[Dict[Any, int]] = None,
    local_rate_lookup: Optional[Dict[str, Dict[int, float]]] = None,
    stack_of_events_per_node: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Any:
    """
    Calculate all placement costs for a given node and projection.

    This function orchestrates the cost calculation process by:
    1. Running initial cost calculation via prePP
    2. Adjusting costs for locally available events
    3. Adding transmission costs to sinks if needed

    Args:
        placement_node: Node ID where projection will be placed
        current_projection: The projection/query being processed
        query_workload: List of queries in the workload
        network_data_nodes: List of network nodes
        pairwise_selectivities: Dictionary of selectivity values
        dependencies_per_projection: Dictionary mapping projections to combinations
        global_event_rates: Event rate dictionary
        projection_rates_selectivity: Projection output rates
        index_event_nodes: Event node index mapping
        simulation_mode: Simulation mode (deterministic/random)
        pairwise_distance_matrix: All-pairs shortest path distances
        sink_nodes: List of sink node IDs
        has_placed_subqueries: Whether subqueries are already placed
        placed_subqueries: Dict mapping subqueries to their placement node IDs

    Returns:
        Tuple: (all_push_costs, push_pull_costs, latency, computing_time,
                transmission_ratio, acquisition_steps)
    """
    logger.debug(
        f"Calculating costs for node {placement_node}, projection {current_projection}"
    )
    logger.debug(
        f"local_rate_lookup parameter in calculate_costs: {local_rate_lookup is not None}"
    )

    selectivity_rate = projection_rates_selectivity.get(current_projection, (0.0, 0.0))[
        0
    ]

    # Initial cost calculation
    results = calculate_prepp_with_placement(
        placement_node=placement_node,
        projection=current_projection,
        query_workload=query_workload,
        network=network_data_nodes,
        selectivity_rate=selectivity_rate,
        selectivities=pairwise_selectivities,
        combination_dict=dependencies_per_projection,
        rates=global_event_rates,
        projection_rates=projection_rates_selectivity,
        index_event_nodes=index_event_nodes,
        mode=simulation_mode,
        shortest_path_distances=pairwise_distance_matrix,
        nodes_per_primitive_event=nodes_per_primitive_event,
        has_placed_subqueries=has_placed_subqueries,
        placed_subqueries=placed_subqueries,
        local_rate_lookup=local_rate_lookup,
    )

    # Adjust costs if some events are already available at the node
    (
        all_push_costs,
        push_pull_costs,
        all_push_latency,
        push_pull_latency,
        transmission_ratio,
    ) = handle_locally_available_events(
        results=results,
        placement_node=placement_node,
        projection=current_projection,
        stack_of_events_per_node=stack_of_events_per_node,
    )

    # Check if we need to send results to sinks (e.g., cloud)
    needs_to_be_sent_to_cloud = check_if_projection_needs_to_be_sent_to_cloud(
        projection=current_projection,
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
                current_latency=push_pull_latency,
                current_transmission_ratio=transmission_ratio,
                placement_node=placement_node,
                projection=current_projection,
                sink_nodes=sink_nodes,
                projection_rates=projection_rates_selectivity,
                shortest_path_distances=pairwise_distance_matrix,
            )
        )

    # Return final costs and metrics (no modifications for global optimization)
    return (
        all_push_costs,
        push_pull_costs,
        all_push_latency,
        push_pull_latency,
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
    results,
    placement_node,
    projection,
    stack_of_events_per_node: Optional[Dict[int, Dict[str, Any]]] = None,
) -> tuple[float, float, float, float, float]:
    """
    Handle cost adjustments for locally available events at placement node.

    Args:
        results: Results from initial cost calculation
        placement_node: Node ID where projection is placed
        projection: The projection being processed
        stack_of_events_per_node: Stack tracking events at each node

    Returns:
        Tuple: (adjusted_all_push_costs, adjusted_push_pull_costs,
                adjusted_all_push_latency, adjusted_push_pull_latency,
                adjusted_transmission_ratio)
    """
    if stack_of_events_per_node is None:
        stack_of_events_per_node = {}

    available_events = set(
        get_events_from_stack(stack_of_events_per_node, placement_node)
    )
    needed_events = get_events_for_projection(projection)

    if available_events & needed_events:  # If there's intersection
        return handle_intersection(
            placement_node,
            results,
            available_events,
            needed_events,
            stack_of_events_per_node,
        )
    else:
        return (
            results["all_push_costs"],
            results["push_pull_costs"],
            results["all_push_latency"],
            results["push_pull_latency"],
            results["transmission_ratio"],
        )


def _adjust_acquisition_steps(
    acquisition_steps: Dict[int, Dict[str, Any]],
    events_already_available: Set[str],
    node: int,
    stack_of_events_per_node: Dict[int, Dict[str, Any]],
) -> Tuple[float, float]:
    """
    Adjust acquisition steps by removing costs for locally available events.

    Logic:
    - If ALL events in a step are available: remove entire step cost and latency
    - If NO events in a step are available: keep step unchanged
    - If SOME events are available: remove their cost contribution, keep latency

    Args:
        acquisition_steps: Dict mapping step number to step details
        events_already_available: Set of events available locally
        node: Node ID
        stack_of_events_per_node: Stack tracking events at each node

    Returns:
        Tuple of (total_cost_adjustment, total_latency_adjustment)
    """
    total_cost_adjustment = 0.0
    total_latency_adjustment = 0.0

    for step_details in acquisition_steps.values():
        events_to_pull = step_details.get("events_to_pull", [])
        if not events_to_pull:
            continue

        # Extract primitive events from this step
        all_primitive_events_in_step = _extract_primitive_events_from_step(
            events_to_pull
        )

        # Find intersection with available events
        events_we_can_skip = all_primitive_events_in_step & events_already_available

        if not events_we_can_skip:
            # No events available, keep step as-is
            continue

        # Update metadata about which events are already at node
        _update_step_metadata(
            step_details, events_we_can_skip, node, stack_of_events_per_node
        )

        # Check if ALL events in step are available
        if all_primitive_events_in_step <= events_already_available:
            # Complete step skip - remove entire cost and latency
            step_cost = step_details.get("total_step_costs", 0.0)
            step_latency = step_details.get("total_latency", 0.0)
            total_cost_adjustment += step_cost
            total_latency_adjustment += step_latency
        else:
            # Partial skip - calculate cost contribution of available events
            cost_adj = _calculate_partial_step_adjustment(
                step_details, events_we_can_skip, all_primitive_events_in_step
            )
            total_cost_adjustment += cost_adj
            # Keep latency as-is for partial steps

    return total_cost_adjustment, total_latency_adjustment


def _adjust_all_push_acquisition(
    push_pull_acquisition_steps: Dict[int, Dict[str, Any]],
    events_already_available: Set[str],
    node: int,
    stack_of_events_per_node: Dict[int, Dict[str, Any]],
) -> Tuple[float, float]:
    """
    Adjust all-push strategy costs when we only have push-pull acquisition steps.

    For all-push, we look at step 0's pull_response to determine which events
    would be directly pulled and their costs.

    Args:
        push_pull_acquisition_steps: Push-pull acquisition steps from PrePP
        events_already_available: Set of events available locally
        node: Node ID
        stack_of_events_per_node: Stack tracking events at each node

    Returns:
        Tuple of (cost_adjustment, latency_adjustment)
    """
    if 0 not in push_pull_acquisition_steps:
        return 0.0, 0.0

    step_0 = push_pull_acquisition_steps[0]
    pull_response = step_0.get("detailed_cost_contribution", {}).get(
        "pull_response", {}
    )

    if not pull_response:
        return 0.0, 0.0

    # Get all events that would be pulled in all-push (from pull_response)
    all_push_events = set(pull_response.keys())

    # Find which of these are already available
    events_we_can_skip = all_push_events & events_already_available

    if not events_we_can_skip:
        return 0.0, 0.0

    total_cost_adjustment = 0.0
    max_latency = 0.0

    # Calculate cost and latency adjustments
    for event in events_we_can_skip:
        if event in pull_response:
            event_details = pull_response[event]
            cost = event_details.get("cost_with_selectivity", 0.0)
            latency = event_details.get("latency", 0)

            total_cost_adjustment += cost
            max_latency = max(max_latency, latency)

    # If ALL events are available, we can remove the latency too
    if all_push_events <= events_already_available:
        return total_cost_adjustment, max_latency
    else:
        # Partial availability - keep latency
        return total_cost_adjustment, 0.0


def _calculate_partial_step_adjustment(
    step_details: Dict[str, Any],
    events_we_can_skip: Set[str],
    all_primitive_events_in_step: Set[str],
) -> float:
    """
    Calculate cost adjustment for partially available events in a step.

    Args:
        step_details: Step details dictionary
        events_we_can_skip: Events that are locally available
        all_primitive_events_in_step: All events needed in this step

    Returns:
        Cost adjustment amount
    """
    pull_response_details = step_details.get("detailed_cost_contribution", {}).get(
        "pull_response", {}
    )

    total_cost_adjustment = 0.0

    # Try to get exact cost for each available event
    for event in events_we_can_skip:
        if event in pull_response_details:
            cost = pull_response_details[event].get("cost_with_selectivity", 0.0)
            total_cost_adjustment += cost
        else:
            # Fallback: proportional adjustment
            fraction = len(events_we_can_skip) / len(all_primitive_events_in_step)
            step_cost = step_details.get("total_step_costs", 0.0)
            total_cost_adjustment += step_cost * fraction
            break  # Only do fallback once for all events

    return total_cost_adjustment


def _extract_primitive_events_from_step(events_to_pull: List[Any]) -> Set[str]:
    """
    Extract all primitive events from a list of events or subqueries in an acquisition step.

    Args:
        events_to_pull: List of events or subqueries to extract primitive events from

    Returns:
        Set of primitive event names
    """
    all_primitive_events = set()
    for event_or_subquery in events_to_pull:
        if isinstance(event_or_subquery, str):
            primitive_events = determine_all_primitive_events_of_projection(
                event_or_subquery
            )
        else:
            primitive_events = determine_all_primitive_events_of_projection(
                str(event_or_subquery)
            )
        all_primitive_events.update(primitive_events)
    return all_primitive_events


def _update_step_metadata(
    step_details: Dict[str, Any],
    events_we_can_skip: Set[str],
    node: int,
    stack_of_events_per_node: Dict[int, Dict[str, Any]],
) -> None:
    """
    Update step details with metadata about acquired events.

    Args:
        step_details: Step details dictionary to update
        events_we_can_skip: Set of events that can be skipped
        node: Node ID
        stack_of_events_per_node: Stack tracking events at each node
    """
    step_details["already_at_node"] = list(events_we_can_skip)
    step_details["acquired_by_query"] = {}

    for event in events_we_can_skip:
        metadata = get_event_metadata_from_stack(stack_of_events_per_node, node, event)
        if metadata:
            step_details["acquired_by_query"][event] = metadata.get("query_id")


def handle_intersection(
    node: int,
    results: Dict[str, Any],
    available_events: Set[str],
    needed_events: Set[str],
    stack_of_events_per_node: Dict[int, Dict[str, Any]],
) -> Tuple[float, float, float, float, float]:
    """
    Handle cost adjustments when some events are already available at the placement node.

    This function adjusts costs for BOTH all-push and push-pull strategies separately
    based on their respective acquisition plans.

    Args:
        node: Node ID where placement is being considered
        results: Cost calculation results dictionary
        available_events: Set of events available at the node
        needed_events: Set of events needed for the projection
        stack_of_events_per_node: Stack tracking events at each node

    Returns:
        Tuple: (adjusted_all_push_costs, adjusted_push_pull_costs,
                adjusted_all_push_latency, adjusted_push_pull_latency,
                adjusted_transmission_ratio)
    """
    # Extract current costs from results
    current_all_push_costs = results["all_push_costs"]
    current_push_pull_costs = results["push_pull_costs"]
    current_all_push_latency = results["all_push_latency"]
    current_push_pull_latency = results["push_pull_latency"]
    current_transmission_ratio = results["transmission_ratio"]
    push_pull_acquisition_steps = results["aquisition_steps"]

    # Find events that are both available and needed (intersection)
    events_already_available = available_events & needed_events

    if not events_already_available:
        # No intersection, return original values
        return (
            current_all_push_costs,
            current_push_pull_costs,
            current_all_push_latency,
            current_push_pull_latency,
            current_transmission_ratio,
        )

    # Determine which scenario we're in
    strategies_are_same = (
        current_all_push_costs == current_push_pull_costs
        and current_all_push_latency == current_push_pull_latency
    )

    if strategies_are_same:
        # Scenario 1: PrePP returned all-push as optimal (same costs/latencies)
        # Only adjust the push-pull acquisition steps (which is the same as all-push)
        adjusted_costs, adjusted_latency = _adjust_acquisition_steps(
            push_pull_acquisition_steps,
            events_already_available,
            node,
            stack_of_events_per_node,
        )

        adjusted_all_push_costs = max(0.0, current_all_push_costs - adjusted_costs)
        adjusted_push_pull_costs = adjusted_all_push_costs
        adjusted_all_push_latency = max(
            0.0, current_all_push_latency - adjusted_latency
        )
        adjusted_push_pull_latency = adjusted_all_push_latency
    else:
        # Scenario 2: Different strategies - adjust each separately
        # Adjust push-pull using its acquisition steps
        push_pull_cost_adj, push_pull_latency_adj = _adjust_acquisition_steps(
            push_pull_acquisition_steps,
            events_already_available,
            node,
            stack_of_events_per_node,
        )

        # For all-push, we need to create synthetic all-push acquisition steps
        # since PrePP didn't provide them (it returned push-pull as better)
        # We'll use step 0 from push_pull_acquisition_steps to extract needed info
        all_push_cost_adj, all_push_latency_adj = _adjust_all_push_acquisition(
            push_pull_acquisition_steps,
            events_already_available,
            node,
            stack_of_events_per_node,
        )

        adjusted_push_pull_costs = max(
            0.0, current_push_pull_costs - push_pull_cost_adj
        )
        adjusted_all_push_costs = max(0.0, current_all_push_costs - all_push_cost_adj)
        adjusted_push_pull_latency = max(
            0.0, current_push_pull_latency - push_pull_latency_adj
        )
        adjusted_all_push_latency = max(
            0.0, current_all_push_latency - all_push_latency_adj
        )

    # Recalculate transmission ratio
    if adjusted_all_push_costs > 0:
        adjusted_transmission_ratio = adjusted_push_pull_costs / adjusted_all_push_costs
    else:
        adjusted_transmission_ratio = current_transmission_ratio

    return (
        adjusted_all_push_costs,
        adjusted_push_pull_costs,
        adjusted_all_push_latency,
        adjusted_push_pull_latency,
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
