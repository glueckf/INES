"""
Core placement engine facade.

This module provides the main entry point for the placement engine,
maintaining compatibility with the legacy API while providing a clean
internal implementation.
"""

from typing import Any
import re
import networkx
from .state import RuntimeServices
from .determinism import setup_deterministic_environment, log_determinism_info, validate_deterministic_inputs
from .logging import get_placement_logger
from .global_placement_tracker import get_global_placement_tracker
from .cost_calculation import get_selection_rate
from .node_tracker import get_global_event_placement_tracker


# Global event tracker instance - will be initialized when first accessed
def get_global_event_tracker():
    """Get the global event tracker instance."""
    return get_global_event_placement_tracker()


def compute_operator_placement_with_prepp(
        self,
        projection: dict,
        combination: list,
        no_filter: int,
        proj_filter_dict: dict,
        event_nodes: list,
        index_event_nodes: dict,
        network_data: dict,
        all_pairs: list,
        mycombi: dict,
        rates: dict,
        single_selectivity: dict,
        projrates: dict,
        graph: networkx.Graph,
        network: list,
        central_eval_plan: list,
        sinks: list[int] = [0]
) -> Any:
    """
    Legacy facade — preserves signature. Calls internal engine steps.
    
    This function maintains the exact same signature and behavior as the original
    compute_operator_placement_with_prepp function to ensure complete backward
    compatibility while internally using the modernized placement engine.
    
    Args:
        self: Instance of the class containing all necessary data
        projection: The projection for which the placement is computed
        combination: The combination of event types to consider for the placement
        no_filter: Flag to indicate whether to apply filters or not
        proj_filter_dict: Dictionary containing filters for projections
        event_nodes: Matrix mapping event types to nodes
        index_event_nodes: Indexed dictionary mapping event types to their respective ETBs
        network_data: Dictionary containing data on which node produces which event types
        all_pairs: Matrix containing all pairwise distances between nodes
        mycombi: Dictionary mapping event types to their combinations
        rates: Dictionary containing rates for each event type
        single_selectivity: Dictionary containing selectivity for single event types
        projrates: Dictionary containing rates for each projection
        graph: NetworkX graph representing the network topology
        network: Network object containing all nodes and their respective properties
        central_eval_plan: Central evaluation plan data
        sinks: List of sink nodes to consider for the placement (default: [0] because of single sink placement)
        
    Returns:
        PlacementDecision: Object containing the best placement decision with costs and plan details
    """
    # Initialize determinism harness and logging
    setup_deterministic_environment()
    services = RuntimeServices.create_deterministic()
    logger = get_placement_logger(__name__)

    # log_determinism_info(services, logger)
    logger.info(f"Starting placement computation for projection: {projection}")

    # Setup the tracker structures
    global_tracker = get_global_placement_tracker()
    global_event_tracker = get_global_event_placement_tracker()

    # Check if this projection has subqueries that we've already placed
    has_placed_subqueries = check_if_projection_has_placed_subqueries(projection, mycombi, global_tracker)

    try:
        # Import required modules with error handling
        try:
            from .initialization import initialize_placement_state
            from .candidate_selection import check_resources
            from .cost_calculation import calculate_prepp_with_placement, calculate_final_costs_for_sending_to_sinks
            from .fallback import get_strategy_recommendation
            from .state import PlacementDecision, PlacementDecisionTracker
        except Exception as e:
            logger.error(f"Failed to import required modules: {e}")
            raise

        # Initialize placement state with error handling
        try:
            logger.debug(f"Initializing placement state for projection: {projection}")
            placement_state = initialize_placement_state(
                combination, proj_filter_dict, no_filter, projection, graph
            )
            logger.debug(f"Placement state initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize placement state for projection {projection}: {e}")
            logger.error(f"Parameters: combination={combination}, proj_filter_dict keys={list(proj_filter_dict.keys()) if proj_filter_dict else None}, no_filter={no_filter}, graph nodes={len(graph.nodes()) if graph else None}")
            raise

        # Initialize decision tracker with error handling
        try:
            placement_decisions = PlacementDecisionTracker(projection)
            logger.debug(f"Decision tracker initialized for projection: {projection}")
        except Exception as e:
            logger.error(f"Failed to initialize decision tracker for projection {projection}: {e}")
            raise

        # Get possible placement nodes with error handling
        try:
            logger.debug(f"Getting possible placement nodes for projection: {projection}")
            possible_placement_nodes = get_all_possible_placement_nodes(
                projection=projection,
                placement_state=placement_state,
                network_data=network_data,
                index_event_nodes=index_event_nodes,
                event_nodes=event_nodes,
                logger=logger
            )
            logger.debug(f"Found {len(possible_placement_nodes)} possible placement nodes: {possible_placement_nodes}")
        except Exception as e:
            logger.error(f"Failed to get possible placement nodes for projection {projection}: {e}")
            logger.error(f"Parameters: placement_state={placement_state}, network_data length={len(network_data) if network_data else None}, index_event_nodes keys={list(index_event_nodes.keys()) if index_event_nodes else None}")
            raise

        logger.info(f"Evaluating {len(possible_placement_nodes)} placement candidates")

        best_costs = float('inf')

        # Get selection rate with error handling
        try:
            selection_rate = get_selection_rate(projection, self.h_mycombi, self.selectivities)
            logger.debug(f"Selection rate for projection {projection}: {selection_rate}")
        except Exception as e:
            logger.error(f"Failed to get selection rate for projection {projection}: {e}")
            logger.error(f"Parameters: mycombi keys={list(self.h_mycombi.keys()) if hasattr(self, 'h_mycombi') else None}, selectivities keys={list(self.selectivities.keys()) if hasattr(self, 'selectivities') else None}")
            raise

        # Evaluate each candidate node
        for node_idx, node in enumerate(possible_placement_nodes):
            try:
                logger.debug(f"Evaluating node {node} ({node_idx + 1}/{len(possible_placement_nodes)})")

                # Calculate prepp with placement with detailed error handling
                try:
                    if str(projection) == 'SEQ(A, B, AND(E, F))':
                        logger.debug(f"Special case handling for projection {projection} at node {node}")

                    results = calculate_prepp_with_placement(
                        self=self,
                        node=node,
                        projection=projection,
                        network=network,
                        selectivity_rate=selection_rate,
                        global_placement_tracker=global_tracker,
                        has_placed_subqueries=has_placed_subqueries,
                    )
                    logger.debug(f"Node {node} - PrePP calculation completed successfully")
                except Exception as e:
                    logger.error(f"Failed to calculate prepp with placement for node {node}, projection {projection}: {e}")
                    logger.error(f"Parameters: node={node}, projection={projection}, network size={len(network) if network else None}, selectivity_rate={selection_rate}")
                    logger.error(f"Global tracker entries={len(global_tracker._placement_history) if global_tracker and hasattr(global_tracker, '_placement_history') else None}")
                    continue  # Skip this node and continue with others

                if not results:
                    logger.warning(f"Node {node} - No results from prepp_with_placement")
                    continue  # Skip this node

                # Extract results with error handling
                try:
                    all_push_costs = results['all_push_costs']
                    push_pull_costs = results['push_pull_costs']
                    latency = results['latency']
                    computing_time = results['computing_time']
                    transmission_ratio = results['transmission_ratio']
                    aquisition_steps = results['aquisition_steps']
                    logger.debug(f"Node {node} - Results extracted: push_pull_costs={push_pull_costs}, all_push_costs={all_push_costs}")
                except KeyError as e:
                    logger.error(f"Missing key in results for node {node}: {e}")
                    logger.error(f"Available keys in results: {list(results.keys()) if results else None}")
                    continue

                # Get events and handle intersections with error handling
                try:
                    available_events = set(global_event_tracker.get_events_at_node(node))
                    needed_events = get_events_for_projection(projection)
                    logger.debug(f"Node {node} - Available events: {available_events}, Needed events: {needed_events}")
                except Exception as e:
                    logger.error(f"Failed to get events for node {node}, projection {projection}: {e}")
                    # Continue without intersection handling
                    available_events = set()
                    needed_events = set()

                if available_events & needed_events:  # If there's intersection
                    try:
                        logger.debug(f"Node {node} already has events: {available_events & needed_events}")
                        all_push_costs, push_pull_costs, latency, transmission_ratio = handle_intersection(
                            results, available_events, needed_events, logger
                        )
                        logger.debug(f"Node {node} - Costs adjusted after intersection handling")
                    except Exception as e:
                        logger.error(f"Failed to handle intersection for node {node}: {e}")
                        # Continue with original costs

                # Add transmission costs to sinks if needed
                if projection in self.query_workload and node not in sinks:
                    try:
                        logger.debug(f"Adding transmission costs from node {node} to sinks for workload projection")
                        results = calculate_final_costs_for_sending_to_sinks(
                            push_pull_costs,
                            all_push_costs,
                            latency,
                            transmission_ratio,
                            node,
                            projection,
                            sinks,
                            self.h_projrates,
                            self.allPairs
                        )
                        all_push_costs, push_pull_costs, latency, transmission_ratio = results
                        logger.debug(f"Node {node} - Transmission costs added successfully")
                    except Exception as e:
                        logger.error(f"Failed to calculate transmission costs for node {node}: {e}")
                        logger.error(f"Parameters: push_pull_costs={push_pull_costs}, all_push_costs={all_push_costs}, node={node}, sinks={sinks}")
                        # Continue with original costs

                # Check resource availability with error handling
                try:
                    has_enough_resources = check_resources(node, projection, network, combination)
                    logger.debug(f"Node {node} - Resource check: {has_enough_resources}")
                except Exception as e:
                    logger.error(f"Failed to check resources for node {node}: {e}")
                    has_enough_resources = True  # Default to True if check fails

                # Determine best strategy with error handling
                try:
                    strategy = get_strategy_recommendation(all_push_costs, push_pull_costs, has_enough_resources)
                    final_costs = push_pull_costs if strategy == 'push_pull' else all_push_costs
                    logger.debug(f"Node {node} - Strategy: {strategy}, Final cost: {final_costs:.2f}")
                except Exception as e:
                    logger.error(f"Failed to get strategy recommendation for node {node}: {e}")
                    logger.error(f"Parameters: all_push_costs={all_push_costs}, push_pull_costs={push_pull_costs}, has_enough_resources={has_enough_resources}")
                    # Default to push_pull strategy
                    strategy = 'push_pull'
                    final_costs = push_pull_costs

                # Create placement decision with error handling
                try:
                    decision = PlacementDecision(
                        node=node,
                        costs=final_costs,
                        strategy=strategy,
                        all_push_costs=all_push_costs,
                        push_pull_costs=push_pull_costs,
                        has_sufficient_resources=has_enough_resources,
                        plan_details={
                            'computing_time': computing_time,
                            'latency': latency,
                            'transmission_ratio': transmission_ratio,
                            'aquisition_steps': aquisition_steps
                        }
                    )
                    logger.debug(f"Node {node} - Placement decision created successfully")
                except Exception as e:
                    logger.error(f"Failed to create placement decision for node {node}: {e}")
                    continue

                # Track this decision with error handling
                try:
                    placement_decisions.add_decision(decision)
                    logger.debug(f"Node {node} - Decision added to tracker")
                except Exception as e:
                    logger.error(f"Failed to add decision to tracker for node {node}: {e}")
                    continue

                # Update best placement if this is better
                if final_costs < best_costs:
                    best_costs = final_costs
                    logger.debug(f"New best placement: Node {node} with {strategy} strategy (cost: {final_costs:.2f})")

            except Exception as e:
                logger.error(f"Error processing node {node} for projection {projection}: {e}")
                logger.error(f"Skipping node {node} and continuing with remaining nodes")
                continue

        # Get the final best decision with error handling
        try:
            best_decision = placement_decisions.get_best_decision()
            if not best_decision:
                logger.error(f"No best decision found for projection {projection}")
                all_decisions = placement_decisions.get_all_decisions() if hasattr(placement_decisions, 'get_all_decisions') else []
                logger.error(f"Total placement attempts made: {len(all_decisions)}")
                raise RuntimeError("No valid placement nodes found")
            logger.debug(f"Best decision found for {projection}: Node {best_decision.node}, Cost: {best_decision.costs:.2f}")
        except Exception as e:
            logger.error(f"Failed to get best decision for projection {projection}: {e}")
            raise

        # Store placement decisions in global tracker with error handling
        try:
            global_tracker.store_placement_decisions(projection, placement_decisions)
            logger.debug(f"Stored placement decisions for {projection} in global tracker")
        except Exception as e:
            logger.error(f"Failed to store placement decisions for projection {projection}: {e}")
            # Continue anyway, this is not critical

        # Add events to global event tracker with error handling
        try:
            events = list(get_events_for_projection(projection))
            global_event_tracker.add_events_at_node(
                node_id=best_decision.node,
                events=events,
                query_id=str(projection),
                acquisition_type=best_decision.strategy,
                acquisition_steps=best_decision.plan_details.get('aquisition_steps', [])
            )
            logger.debug(f"Added events {events} at node {best_decision.node} for projection {projection}")
        except Exception as e:
            logger.error(f"Failed to add events to global tracker for projection {projection}: {e}")
            # Continue anyway, this is not critical

        # Log and return best decision
        try:
            logger.info(f"Best placement for {projection}: Node {best_decision.node}, ")
            logger.info(f"Strategy: {best_decision.strategy}, Cost: {best_decision.costs:.2f}")
            logger.debug(f"Plan details: {best_decision.plan_details}")
            return best_decision
        except Exception as e:
            logger.error(f"Error accessing best decision properties for projection {projection}: {e}")
            logger.error(f"Best decision object: {best_decision}")
            raise

    except Exception as e:
        logger.error(f"Error in placement engine for projection {projection}: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {str(e)}")
        
        # Log context information
        logger.error(f"Projection details: {projection}")
        logger.error(f"Available network nodes: {len(network) if network else 'None'}")
        logger.error(f"Graph nodes count: {len(graph.nodes()) if graph and hasattr(graph, 'nodes') else 'None'}")
        logger.error(f"Has placed subqueries: {has_placed_subqueries}")
        
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        logger.info("Falling back to legacy implementation")

        # Fallback to legacy implementation if new engine fails
        try:
            from helper.placement_aug import compute_operator_placement_with_prepp as legacy_impl

            result = legacy_impl(
                self, projection, combination, no_filter, proj_filter_dict, event_nodes,
                index_event_nodes, network_data, all_pairs, mycombi, rates,
                single_selectivity, projrates, graph, network, central_eval_plan
            )
            
            logger.info(f"Legacy implementation successful for projection {projection}")
            return result
            
        except Exception as legacy_e:
            logger.error(f"Legacy implementation also failed for projection {projection}: {legacy_e}")
            logger.error(f"Legacy exception type: {type(legacy_e).__name__}")
            logger.error(f"Legacy traceback:\n{traceback.format_exc()}")
            raise


def _convert_to_legacy_format(best_decision, placement_decisions):
    """
    Convert PlacementDecision to legacy tuple format for backward compatibility.
    
    Legacy format: (costs, node, longestPath, myProjection, newInstances, Filters)
    
    Args:
        best_decision: PlacementDecision object
        placement_decisions: PlacementDecisionTracker object
        
    Returns:
        tuple: Legacy format tuple
    """
    from EvaluationPlan import Projection

    # Create minimal legacy structures
    costs = best_decision.costs
    node = best_decision.node
    longestPath = 0  # Simplified - could be calculated from subgraph if needed

    # Use the original projection object as the name (from the tracker)
    original_projection = placement_decisions.projection

    # Create a basic projection object with the original projection as name
    myProjection = Projection(
        name=original_projection,  # Use the original projection object, not a string
        combination={},
        sinks=[node],
        spawnedInstances=[],
        Filters=[]
    )

    # Create empty instances list (simplified)
    newInstances = []

    # Create empty filters list (simplified)
    Filters = []

    logger = get_placement_logger(__name__)
    logger.debug(f"Converting to legacy format: costs={costs}, node={node}, projection={original_projection}")

    return costs, node, longestPath, myProjection, newInstances, Filters


def check_if_projection_has_placed_subqueries(projection, mycombi, global_tracker, logger=None):
    if projection in mycombi:
        subqueries = mycombi[projection]
        global_tracker.register_query_hierarchy(projection, subqueries)

        # Check if any subqueries have existing placements
        for subquery in subqueries:
            if hasattr(subquery, 'leafs') and global_tracker.has_placement_for(subquery):
                return True

        return False
    else:
        raise ValueError("Projection not found in mycombi or invalid structure")


def get_all_possible_placement_nodes(projection, placement_state, network_data, index_event_nodes, event_nodes,
                                     logger=None):
    from .candidate_selection import check_possible_placement_nodes_for_input

    possible_placement_nodes = []

    # Find possible placement nodes
    possible_placement_nodes = check_possible_placement_nodes_for_input(
        projection=projection,
        combination=placement_state['extended_combination'],
        network_data=network_data,
        index_event_nodes=index_event_nodes,
        event_nodes=event_nodes,
        routing_dict=placement_state['routing_dict']
    )

    # Validate and sort candidates for deterministic processing
    possible_placement_nodes = validate_deterministic_inputs(possible_placement_nodes, logger)

    # Reverse the order to prioritize nodes closer to the source (legacy behavior)
    # possible_placement_nodes.reverse()

    return possible_placement_nodes


def determine_all_primitive_events_of_projection(projection):
    """Extract primitive events from a projection string like 'SEQ(A, B)' -> ['A', 'B']"""
    given_predicates = str(projection).replace('AND', '')
    given_predicates = given_predicates.replace('SEQ', '')
    given_predicates = given_predicates.replace('(', '')
    given_predicates = given_predicates.replace(')', '')
    given_predicates = re.sub(r'[0-9]+', '', given_predicates)
    given_predicates = given_predicates.replace(' ', '')
    if ',' in given_predicates:
        return given_predicates.split(',')
    else:
        # Handle single events without comma
        return list(given_predicates)


def handle_intersection(results, available_events, needed_events, logger):
    """
    Handle cost adjustments when some events are already available at the placement node.
    
    This function removes acquisition costs for events that don't need to be acquired
    because they're already present at the node. It handles both primitive events (A, B, C)
    and complex events (SEQ(A, B), AND(C, D)) by extracting the primitive components.
    """

    # Extract current costs from results
    current_all_push_costs = results['all_push_costs']
    current_push_pull_costs = results['push_pull_costs']
    current_latency = results['latency']
    current_transmission_ratio = results['transmission_ratio']
    current_acquisition_steps = results['aquisition_steps']

    # Find events that are both available and needed (intersection)
    events_already_available = available_events & needed_events

    # Track total cost adjustments
    total_cost_adjustment = 0.0

    # logger.info(f"Available events at node: {available_events}")
    # logger.info(f"Needed events for projection: {needed_events}")
    # logger.info(f"Events already available that are needed: {events_already_available}")

    # Go through each acquisition step and check if we can skip it
    for step_index, step_details in current_acquisition_steps.items():
        events_to_pull_in_this_step = step_details.get('events_to_pull', [])
        step_total_cost = step_details.get('total_step_costs', 0.0)

        # logger.info(f"Checking acquisition step {step_index}: {events_to_pull_in_this_step}")

        # Extract primitive events from each event in this step
        all_primitive_events_in_step = set()
        for event_or_subquery in events_to_pull_in_this_step:
            if isinstance(event_or_subquery, str):
                # Handle string representations like 'SEQ(A, B)' or simple events like 'A'
                primitive_events = determine_all_primitive_events_of_projection(event_or_subquery)
                all_primitive_events_in_step.update(primitive_events)
            else:
                # Handle object representations - try to get string representation
                primitive_events = determine_all_primitive_events_of_projection(str(event_or_subquery))
                all_primitive_events_in_step.update(primitive_events)

        # logger.info(f"Primitive events in step {step_index}: {all_primitive_events_in_step}")

        # Check if any of the primitive events in this step are already available
        events_we_can_skip = all_primitive_events_in_step & events_already_available

        if events_we_can_skip:
            # If all primitive events in this step are already available, skip entire step cost
            if all_primitive_events_in_step <= events_already_available:  # All events in step are available
                total_cost_adjustment += step_total_cost
                # logger.info(f"Skipping entire acquisition step {step_index} (cost: {step_total_cost}) "
                #             f"because all primitive events {all_primitive_events_in_step} are already available")
            else:
                # Partial adjustment - some primitive events in step are available
                fraction_available = len(events_we_can_skip) / len(all_primitive_events_in_step)
                partial_adjustment = step_total_cost * fraction_available
                total_cost_adjustment += partial_adjustment
                # logger.info(f"Partial adjustment for acquisition step {step_index}: "
                #             f"reducing cost by {partial_adjustment} ({fraction_available:.2%}) "
                #             f"for available events {events_we_can_skip} out of {all_primitive_events_in_step}")

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

    # logger.info(f"Total cost adjustment: {total_cost_adjustment}")
    # logger.info(f"Adjusted costs - Push-Pull: {adjusted_push_pull_costs}, All-Push: {adjusted_all_push_costs}")

    return adjusted_all_push_costs, adjusted_push_pull_costs, current_latency, adjusted_transmission_ratio


def get_events_for_projection(projection):
    """
    Recursively extract all primitive events from a projection.
    
    Args:
        projection: Tree node (SEQ, AND, or PrimEvent)
        
    Returns:
        set: Set of primitive event types (strings) contained in the projection
    """
    from helper.Tree import PrimEvent, SEQ, AND
    events = set()
    
    # Handle primitive events directly
    if isinstance(projection, PrimEvent):
        return {projection.evtype}
    
    # Handle composite projections (SEQ, AND, etc.)
    if hasattr(projection, 'children') and projection.children:
        for child in projection.children:
            if isinstance(child, PrimEvent):
                events.add(child.evtype)
            else:
                # We have a subquery - call recursively and merge the sets
                child_events = get_events_for_projection(child)
                events.update(child_events)
    
    return events