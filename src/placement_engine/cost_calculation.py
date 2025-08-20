"""
Cost calculation for all-push and push-pull strategies.

This module handles the cost calculation logic for both placement
strategies, coordinating with adapters for legacy function calls.
"""

from typing import Dict, List, Any, Optional
import re
from helper.structures import getNodes
from allPairs import find_shortest_path_or_ancestor
from helper.placement_aug import NEWcomputeCentralCosts
from .adapters import build_eval_plan, run_prepp
from .logging import get_placement_logger
from .fallback import calculate_fallback_costs

# Import SimulationMode for deterministic mode checking
try:
    from INES import SimulationMode
except ImportError:
    # Fallback if import fails
    class SimulationMode:
        FULLY_DETERMINISTIC = "deterministic"

logger = get_placement_logger(__name__)


def calculate_all_push_costs_on_subgraph(self, subgraph: Dict[str, Any], projection: Any):
    """
    Calculate the costs of using all-push strategy for a projection on a subgraph.
    This is an efficient implementation that reuses logic from push_pull_plan_generator.
    
    Args:
        subgraph: Dictionary containing subgraph information
        projection: The projection being placed
        
    Returns:
        float: Total cost of all-push strategy
    """
    return NEWcomputeCentralCosts(
        workload=[projection],
        IndexEventNodes=subgraph['index_event_nodes_sub'],
        allPairs=subgraph['all_pairs_sub'],
        rates=self.h_rates_data,
        EventNodes=subgraph['event_nodes_sub'],
        G=subgraph['subgraph']
    )


def calculate_prepp_costs_on_subgraph(self, node: int, subgraph: Dict[str, Any], projection: Any,
                                      central_eval_plan: Any, routing_algo: Any,
                                      all_push_baseline: Optional[float] = 6278.0,
                                      ) -> float:
    """
    Calculate prepp costs on the extracted subgraph by generating evaluation plan and calling prePP.
    Enhanced version that uses all_push_baseline for comparison.
    
    Args:
        self: Instance with network data
        node: Original placement node ID
        subgraph: Subgraph information dictionary
        projection: Projection being placed
        central_eval_plan: Central evaluation plan
        all_push_baseline: Pre-calculated all-push costs for comparison
    
    Returns:
        float: Push-pull strategy cost
    """
    logger.info(f"Calculating push-pull costs for node {node}")
    if all_push_baseline:
        logger.info(f"All-push baseline: {all_push_baseline:.2f}")

    try:
        # Create evaluation plan structures (simplified)
        subgraph_plan = _create_basic_evaluation_plan(self, subgraph, projection, routing_algo)
        central_plan_subgraph = central_eval_plan

        # Generate evaluation plan using adapter
        print("Debug hook")
        eval_plan_buffer = build_eval_plan(
            nw=subgraph['sub_network'],
            selectivities=self.selectivities,
            my_plan=[subgraph_plan, 12345, {}],  # Format: [plan, ID, dict]
            central_plan=central_plan_subgraph,  # Format: [source, dict, workload]
            workload=[projection]  # Single projection as workload
        )

        content = eval_plan_buffer.getvalue()

        # Check if we should use deterministic behavior
        is_deterministic = getattr(self, 'config', None) and getattr(self.config, 'mode', None) == SimulationMode.FULLY_DETERMINISTIC
        
        # Call generate_prePP using adapter
        prepp_results = run_prepp(
            input_buffer=eval_plan_buffer,
            method="ppmuse",
            algorithm="e",  # exact
            samples=0,
            top_k=0,
            runs=1,
            plan_print=True,
            all_pairs=subgraph['all_pairs_sub'],
            is_deterministic=is_deterministic
        )

        # Extract costs from prePP results
        if prepp_results and len(prepp_results) > 0:
            costs = prepp_results[0]  # exact_cost
            exec_time = prepp_results[1] if len(prepp_results) > 1 else 0
            latency = prepp_results[2] if len(prepp_results) > 2 else 0
            transmission_ratio = prepp_results[3] if len(prepp_results) > 3 else 0
            central_costs = prepp_results[4] if len(prepp_results) > 4 else 0
            
            # ===========================================
            # PREPP RESULT LOGGING FOR NODE
            # ===========================================
            print(f"\n{'='*50}")
            print(f"PREPP RESULT FOR NODE {node}")
            print(f"{'='*50}")
            print(f"Projection: {projection}")
            print(f"Push-Pull Cost: {costs:.4f}")
            print(f"All-Push Baseline: {all_push_baseline:.4f}" if all_push_baseline else "All-Push Baseline: N/A")
            print(f"Central Costs: {central_costs:.4f}")
            print(f"Transmission Ratio: {transmission_ratio:.4f}")
            print(f"Execution Time: {exec_time:.4f} seconds")
            print(f"Latency: {latency}")
            
            if all_push_baseline:
                savings = all_push_baseline - costs
                savings_pct = (savings / all_push_baseline * 100) if all_push_baseline > 0 else 0
                print(f"Cost Savings: {savings:.4f} ({savings_pct:.1f}%)")
                print(f"Strategy Recommendation: {'Push-Pull' if costs < all_push_baseline else 'All-Push'}")
            print(f"{'='*50}\n")
            
            logger.info(f"PrePP costs calculated: {costs:.2f}")

            if all_push_baseline:
                savings = all_push_baseline - costs
                logger.info(f"Savings vs all-push: {savings:.2f} ({(savings / all_push_baseline * 100):.1f}%)")

            return costs
        else:
            logger.warning("No valid prePP results returned, using fallback")
            return calculate_fallback_costs(node, subgraph, projection, all_push_baseline)

    except Exception as e:
        logger.error(f"Error calculating prePP costs: {e}")
        # Fallback to simple cost calculation
        return calculate_fallback_costs(node, subgraph, projection, all_push_baseline)


def _extract_event_types_from_projection(projection: Any) -> List[str]:
    """
    Extract event types from a projection.
    
    Args:
        projection: The projection object or string
    
    Returns:
        List of event type strings
    """
    if hasattr(projection, 'leafs'):
        return projection.leafs()

    # Fallback: try to extract from string representation
    projection_str = str(projection)
    # Extract capital letters that represent event types
    event_types = re.findall(r'[A-Z]', projection_str)
    return list(set(event_types))  # Remove duplicates


def _create_basic_evaluation_plan(self, subgraph: Dict[str, Any], projection: Any, routing_algo: Any) -> Any:
    """Create a basic evaluation plan structure for the subgraph."""
    from EvaluationPlan import EvaluationPlan, Projection, Instance

    # Create empty evaluation plan
    evaluation_plan = EvaluationPlan([], [])

    # Initialize instances for primitive events in the subgraph
    evaluation_plan.initInstances(subgraph['index_event_nodes_sub'])

    # Remap the placement node to the subgraph's placement node
    remapped_placement_node = subgraph['placement_node_remapped']

    # Create a relevant projection for the evaluation plan
    current_projection = Projection(
        name=projection,
        combination={},
        sinks=[remapped_placement_node],
        spawnedInstances=[],
        Filters=[])

    # Extract event types from projection
    event_types = _extract_event_types_from_projection(projection)

    # Add instances to the projection based on the subgraph's event nodes
    for event_type in event_types:
        if event_type in subgraph['index_event_nodes_sub']:
            instances_for_event = []

            for etb in subgraph['index_event_nodes_sub'][event_type]:
                # Get source nodes for this ETB in the original graph
                original_sources = getNodes(etb, self.h_eventNodes, self.h_IndexEventNodes)

                # Find corresponding sources in subgraph
                subgraph_sources = []
                for orig_source in original_sources:
                    if orig_source in subgraph['node_mapping']:
                        subgraph_sources.append(subgraph['node_mapping'][orig_source])

                if subgraph_sources:
                    # Create routing info - use simple direct path for now

                    routing_dict = find_shortest_path_or_ancestor(
                        routingDict=routing_algo,
                        me=subgraph_sources[0],
                        j=remapped_placement_node)

                    # Create instance for this ETB
                    instance = Instance(
                        name=event_type,
                        projname=etb,
                        sources=subgraph_sources,
                        routingDict={projection: routing_dict}
                    )
                    instances_for_event.append(instance)

            if instances_for_event:
                current_projection.addInstances(event_type, instances_for_event)

    # Add the projection to the evaluation plan
    evaluation_plan.addProjection(current_projection)

    return evaluation_plan


def _create_basic_central_plan(self, subgraph: Dict[str, Any], central_eval_plan: Any) -> List[Any]:
    """Create a basic central plan structure for the subgraph."""
    # Simplified central plan - use first node as source
    source_node = 0 if subgraph['sub_network'] else 0
    return [source_node, {}, []]


def calculate_final_costs_for_sending_to_sinks(
        cost_results: tuple,
        placement_node: int,
        query_projection: Any,
        sink_nodes: List[int],
        projection_rates: Dict[Any, tuple],
        shortest_path_distances: Dict[int, Dict[int, int]]
) -> tuple:
    """
    Calculate the final transmission costs for sending query results to sink nodes.
    
    This function takes initial push-pull costs and adds the network transmission
    costs to reach all specified sink nodes, factoring in the output rate of the
    projection and the shortest path distances.
    
    Args:
        cost_results: Tuple containing (push_pull_costs, original_result_at_idx_1, 
                     latency, transmission_ratio, all_push_costs)
        placement_node: Node ID where the projection is placed
        query_projection: The projection/query being processed
        sink_nodes: List of destination node IDs to send results to
        projection_rates: Dictionary mapping projections to their rate tuples
        shortest_path_distances: All-pairs shortest path distance matrix
        
    Returns:
        tuple: Updated costs tuple with transmission costs to sinks included,
               or original results if costs cannot be calculated
    """
    print("Hook")
    (push_pull_costs, original_result_at_idx_1, latency,
     transmission_ratio, all_push_costs) = cost_results

    # Extract the output rate for the given projection
    projection_output_rate = (projection_rates[query_projection][1]
                              if isinstance(projection_rates, dict) else 1)

    # Calculate hop distances from placement node to all sink nodes
    hop_distances_to_sinks = [shortest_path_distances[placement_node][sink]
                              for sink in sink_nodes]

    # Log transmission planning information
    logger.info(f"Output rate for projection {query_projection}: {projection_output_rate}")
    for sink in sink_nodes:
        logger.info(f"Distance from node {placement_node} to sink {sink}: "
                    f"{shortest_path_distances[placement_node][sink]}")

    # Calculate final costs including transmission to sinks
    if push_pull_costs is not None and hop_distances_to_sinks:
        # Sum all hop distances and multiply by projection output rate
        total_transmission_cost = sum(hop_distances_to_sinks) * projection_output_rate

        # Add transmission costs to existing strategy costs
        final_push_pull_costs = push_pull_costs + total_transmission_cost
        final_all_push_costs = all_push_costs + total_transmission_cost

        # Latency is dominated by the longest path to any sink
        final_latency = latency + max(hop_distances_to_sinks)

        # Recalculate transmission efficiency ratio
        final_transmission_ratio = final_push_pull_costs / final_all_push_costs

        return (
            final_push_pull_costs,
            original_result_at_idx_1,
            final_latency,
            final_transmission_ratio,
            final_all_push_costs
        )
    else:
        return cost_results


def calculate_prepp_with_placement(
        self,
        node: int,
        projection: Any,
        network,
        selectivity_rate: float,
        global_placement_tracker,
        has_placed_subqueries: bool = False,
):

    input_buffer = initiate_buffer(
        node, 
        projection, 
        network, 
        self.selectivities, 
        selectivity_rate, 
        global_placement_tracker=global_placement_tracker,
        has_placed_subqueries=has_placed_subqueries,
        mycombi=getattr(self, 'h_mycombi', {}),
        rates=getattr(self, 'h_rates_data', {}),
        projrates=getattr(self, 'h_projrates', {}),
        index_event_nodes=getattr(self, 'h_IndexEventNodes', {})
    )

    content = input_buffer.getvalue()

    # Select method and algorithm type for prePP
    method = "ppmuse"
    algorithm = "e"

    # Check if we should use deterministic behavior
    is_deterministic = getattr(self, 'config', None) and getattr(self.config, 'mode', None) == SimulationMode.FULLY_DETERMINISTIC
    
    results = run_prepp(
        input_buffer=input_buffer,
        method=method,
        algorithm=algorithm,
        samples=0,
        top_k=0,
        runs=1,
        plan_print=True,
        all_pairs=self.allPairs,
        is_deterministic=is_deterministic)

    push_pull_costs = results[0]
    logger.info(f"Calculated push-pull costs for projection {projection}: {push_pull_costs:.2f}")

    computing_time = results[1]
    logger.info(f"Computing time for prePP: {computing_time:.2f}")

    # TODO: Discuss latency output because it seems fishy
    latency = results[2] - 1
    logger.info(f"Latency for prePP: {latency:.2f}")

    transmission_ratio = results[3]
    logger.info(f"Transmission ratio for prePP: {transmission_ratio:.2f}")

    # TODO: Discuss all_push_costs.
    #  PrePP adds 1 to all_push_costs because it assumes, that the costs are 1 after evaluating query
    all_push_costs = results[4]
    logger.info(f"All-push costs for projection {projection}: {all_push_costs:.2f}")

    return push_pull_costs, computing_time, latency, transmission_ratio, all_push_costs


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
    index_event_nodes=None
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
    if not _validate_buffer_inputs(node, projection, network, selectivities, selection_rate):
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
            combination_keys = _extract_combination_keys(projection)
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
            global_placement_tracker=global_placement_tracker,
            mycombi=mycombi,
            projrates=projrates
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
        logger.warning(f"Invalid selectivities: expected dict, got {type(selectivities)}")
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
    filters = {}
    if hasattr(projection, 'Filters'):
        for filter_tuple in projection.Filters:
            if len(filter_tuple) >= 2:
                filters[filter_tuple[0]] = filter_tuple[1]
    return filters


def _extract_combination_keys(projection):
    """Extract combination keys from projection."""
    if hasattr(projection, 'combination') and hasattr(projection.combination, 'keys'):
        return list(map(str, projection.combination.keys()))
    elif hasattr(projection, 'children'):
        return list(map(str, projection.children))
    else:
        return [str(projection)]


def _extract_projection_sinks(projection, default_node):
    """Extract sink nodes from projection or use default."""
    if hasattr(projection, 'sinks') and projection.sinks:
        return projection.sinks
    else:
        return [default_node]


def _process_forwarding_dict(projection, forwarding_dict):
    """Process forwarding dictionary for the projection (simplified)."""
    # Simplified version - would need actual instance processing logic
    if hasattr(projection, 'combination'):
        # This would contain the actual forwarding logic from the original
        pass
    return forwarding_dict


def _generate_plan_content(network, selectivities, workload, combination_dict, sink_dict, selection_rate_dict, 
                          projection=None, has_placed_subqueries=False, global_placement_tracker=None, 
                          mycombi=None, projrates=None):
    """Generate the actual plan content string."""
    lines = []

    # Convert Node objects to node IDs
    def extract_node_ids(node_list):
        if node_list is None:
            return None
        if not hasattr(node_list, '__iter__') or isinstance(node_list, str):
            return node_list

        ids = []
        for item in node_list:
            if hasattr(item, 'id'):
                ids.append(item.id)
            elif hasattr(item, 'nodeID'):
                ids.append(item.nodeID)
            elif hasattr(item, 'node_id'):
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
                except:
                    ids.append(str(item))
        return ids if ids else None

    # First pass: collect all children relationships to build parent map
    parent_map = {}  # child_id -> [parent_ids]
    all_children = {}  # node_id -> [child_ids]

    for i, node in enumerate(network):
        children_raw = (getattr(node, 'children', None) or
                        getattr(node, 'child', None) or
                        getattr(node, 'Child', None) or
                        getattr(node, 'childs', None))

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
        comp_power = (getattr(node, 'computational_power', None) or
                      getattr(node, 'computationalPower', None) or
                      getattr(node, 'comp_power', None) or
                      'inf')

        # Try different possible attribute names for memory  
        memory = (getattr(node, 'memory', None) or
                  getattr(node, 'Memory', None) or
                  'inf')

        # Try different possible attribute names for event rates
        event_rates = (getattr(node, 'event_rates', None) or
                       getattr(node, 'eventrates', None) or
                       getattr(node, 'Eventrates', None) or
                       getattr(node, 'rates', None) or
                       [0] * 6)

        # Get parents from the parent map we built
        parents = parent_map.get(i, None)

        # Get children 
        children = all_children[i]

        # Try different possible attribute names for siblings
        siblings_raw = (getattr(node, 'siblings', None) or
                        getattr(node, 'Siblings', None) or
                        getattr(node, 'sibling', None))
        siblings = extract_node_ids(siblings_raw)

        lines.append(f"Computational Power: {comp_power}")
        lines.append(f"Memory: {memory}")
        lines.append(
            f"Eventrates: {list(event_rates) if hasattr(event_rates, '__iter__') and not isinstance(event_rates, str) else event_rates}")
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
        if has_placed_subqueries and projection and mycombi and global_placement_tracker and projrates:
            # Get subqueries for this projection
            if projection in mycombi:
                subqueries = mycombi[projection]
                
                # Add placed subqueries first
                for subquery in subqueries:
                    if hasattr(subquery, 'leafs') and global_placement_tracker.has_placement_for(subquery):
                        placement_decision = global_placement_tracker.get_best_placement(subquery)
                        subquery_rate = projrates.get(subquery, 0.001)
                        if len(subquery_rate) > 1:
                            subquery_rate = subquery_rate[0]
                        
                        # Get primitive events for subquery
                        if hasattr(subquery, 'leafs'):
                            primitive_events = subquery.leafs()
                        else:
                            primitive_events = [str(subquery)]
                        
                        combination_str = "; ".join(primitive_events)
                        lines.append(
                            f"SELECT {subquery} FROM {combination_str} ON {{{placement_decision.node}}} WITH selectionRate= {subquery_rate}")
                
                # Add main query with virtual events (replace placed subqueries with their virtual names)
                main_query = workload[0]
                main_rate = selection_rate_dict.get(main_query, 0)
                
                # Create combination for main query
                main_combination = []
                for elem in subqueries:
                    if hasattr(elem, 'leafs') and global_placement_tracker.has_placement_for(elem):
                        # This subquery is placed, so we reference it as virtual event
                        main_combination.append(str(elem))
                    else:
                        # This is a primitive event
                        main_combination.append(str(elem))
                
                combination_str = "; ".join(main_combination)
                lines.append(
                    f"SELECT {main_query} FROM {combination_str} ON {{0}} WITH selectionRate= {main_rate}")
            else:
                # Fallback to normal handling
                _add_normal_muse_graph_entry(lines, workload, selection_rate_dict, sink_dict, combination_dict)
        else:
            # Normal case without subqueries
            _add_normal_muse_graph_entry(lines, workload, selection_rate_dict, sink_dict, combination_dict)

    return "\n".join(lines)


def _add_normal_muse_graph_entry(lines, workload, selection_rate_dict, sink_dict, combination_dict):
    """Add normal muse graph entry without subqueries."""
    query = workload[0]
    rate = selection_rate_dict.get(query, 0)
    sink_info = sink_dict.get(query, [[], ""])
    sink_nodes = sink_info[0] if sink_info[0] else [0]
    
    # Get combination for query - extract primitive events
    if hasattr(query, 'leafs'):
        primitive_events = query.leafs()
    else:
        primitive_events = combination_dict.get(query, [query])
    
    combination_str = "; ".join(str(e) for e in primitive_events)
    
    lines.append(
        f"SELECT {query} FROM {combination_str} ON {{{', '.join(map(str, sink_nodes))}}} WITH selectionRate= {rate}")


def get_selection_rate(projection: Any, combination_dict, selectivities):
    """
    Calculate selection rate for a projection based on combination dictionary and selectivities.
    
    For queries with subqueries like AND(D, AND(A,B)), this function expands the subqueries
    and creates all possible combinations across the primitive events (e.g., AD, AB, BD).
    
    Args:
        projection: The projection to calculate selection rate for
        combination_dict: Dictionary mapping projections to their event type combinations
        selectivities: Dictionary mapping event type combinations to their selectivity values
        
    Returns:
        float: The calculated selection rate (product of relevant selectivities)
    """

    # Safety check: ensure projection exists in combination_dict
    if projection not in combination_dict:
        logger.warning(f"Projection {projection} not found in combination_dict")
        return 1.0  # Return neutral value

    combination_for_given_projection = combination_dict[projection]

    # Safety check: ensure combination is not empty
    if not combination_for_given_projection:
        logger.warning(f"Empty combination for projection {projection}")
        return 1.0

    from itertools import combinations

    # Check if combination has subqueries
    has_subquery = any(elem in combination_dict.keys() for elem in combination_for_given_projection)
    
    if has_subquery:
        # Expand all elements to their primitive events
        expanded_events = []
        for elem in combination_for_given_projection:
            if elem in combination_dict:
                # This is a subquery, recursively expand it
                sub_events = _expand_to_primitives(elem, combination_dict)
                expanded_events.extend(sub_events)
            else:
                # This is already a primitive event
                expanded_events.append(str(elem))
        
        # Remove duplicates while preserving order
        expanded_events = list(dict.fromkeys(expanded_events))
        logger.info(f"Expanded {projection} to primitive events: {expanded_events}")
        
        # Generate all possible combinations of primitive events (2 to n)
        event_combinations = []
        for r in range(2, len(expanded_events) + 1):
            for combo in combinations(expanded_events, r):
                combo_str = "".join(combo)
                event_combinations.append(combo_str)
                
    else:
        # No subqueries, use original logic
        event_combinations = []
        for r in range(2, len(combination_for_given_projection) + 1):
            for combo in combinations(combination_for_given_projection, r):
                combo_str = "".join(str(element) for element in combo)
                event_combinations.append(combo_str)

    # Calculate selection rate as product of relevant selectivities
    res = 1.0
    for combo_str in event_combinations:
        if combo_str in selectivities:
            selectivity_value = selectivities[combo_str]
            # Safety check: ensure selectivity is a valid number
            if isinstance(selectivity_value, (int, float)) and selectivity_value > 0:
                res *= selectivity_value
                logger.debug(f"Applied selectivity for {combo_str}: {selectivity_value}")
            else:
                logger.warning(f"Invalid selectivity value for {combo_str}: {selectivity_value}")
        else:
            logger.warning(f"Selectivity not found for combination: {combo_str}")

    logger.info(f"Final selection rate for {projection}: {res}")
    return res


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




def update_selectivity(self, projection, selection_rate):
    self.selectivities[str(projection)] = selection_rate
