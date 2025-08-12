"""
Cost calculation for all-push and push-pull strategies.

This module handles the cost calculation logic for both placement
strategies, coordinating with adapters for legacy function calls.
"""

from typing import Dict, List, Any, Optional
from helper.structures import getNodes
from .adapters import build_eval_plan, run_prepp
from .logging import get_placement_logger
from .fallback import calculate_fallback_costs

logger = get_placement_logger(__name__)


def calculate_all_push_costs_on_subgraph(subgraph: Dict[str, Any], projection: Any, 
                                       combination: List[Any], rates: Dict[str, float], 
                                       projrates: Dict[str, List[float]], 
                                       index_event_nodes: Dict[str, List[Any]], 
                                       event_nodes: List[List[Any]]) -> float:
    """
    Calculate the costs of using all-push strategy for a projection on a subgraph.
    This is an efficient implementation that reuses logic from push_pull_plan_generator.
    
    Args:
        subgraph: Dictionary containing subgraph information
        projection: The projection being placed
        combination: List of event types in the combination
        rates: Dictionary of event type rates
        projrates: Dictionary of projection rates
        index_event_nodes: Mapping of event types to ETBs
        event_nodes: Event nodes matrix
        
    Returns:
        float: Total cost of all-push strategy
    """
    logger.debug("Calculating all-push costs on subgraph")
    
    total_cost = 0.0
    placement_node_remapped = subgraph['placement_node_remapped']
    all_pairs = subgraph['all_pairs_sub']
    
    logger.debug(f"All-push calculation for placement at node {placement_node_remapped}")
    
    # Calculate cost for each event type in the combination
    for event_type in combination:
        if event_type in index_event_nodes:
            event_cost = 0.0
            
            for etb in index_event_nodes[event_type]:
                # Get source nodes for this ETB in the original graph
                original_sources = getNodes(etb, event_nodes, index_event_nodes)
                
                # Find corresponding sources in subgraph
                subgraph_sources = []
                for orig_source in original_sources:
                    if orig_source in subgraph['node_mapping']:
                        subgraph_sources.append(subgraph['node_mapping'][orig_source])
                
                if subgraph_sources:
                    # Find best (closest) source
                    best_source = subgraph_sources[0]
                    min_distance = all_pairs[placement_node_remapped][best_source]
                    
                    for source in subgraph_sources[1:]:
                        distance = all_pairs[placement_node_remapped][source]
                        if distance < min_distance:
                            min_distance = distance
                            best_source = source
                    
                    # Calculate cost for this ETB
                    if len(event_type) == 1:  # Primitive event
                        rate = rates.get(event_type, 1.0)
                    else:  # Complex event/projection
                        rate = projrates.get(event_type, [0, 1.0])[1]
                    
                    etb_cost = rate * min_distance
                    event_cost += etb_cost
                    
                    logger.debug(f"Event {event_type}, ETB {etb}: source {best_source} -> sink {placement_node_remapped}, distance {min_distance}, rate {rate:.3f}, cost {etb_cost:.2f}")
            
            total_cost += event_cost
            logger.debug(f"Total cost for event {event_type}: {event_cost:.2f}")
    
    logger.info(f"Total all-push cost: {total_cost:.2f}")
    return total_cost


def calculate_prepp_costs_on_subgraph(self, node: int, subgraph: Dict[str, Any], projection: Any, 
                                    central_eval_plan: Any, all_push_baseline: Optional[float] = None) -> float:
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
        subgraph_plan = _create_basic_evaluation_plan(self, subgraph, projection)
        central_plan_subgraph = _create_basic_central_plan(self, subgraph, central_eval_plan)
        
        # Generate evaluation plan using adapter
        eval_plan_buffer = build_eval_plan(
            nw=subgraph['sub_network'],
            selectivities=self.selectivities,
            my_plan=[subgraph_plan, 12345, {}],  # Format: [plan, ID, dict]
            central_plan=central_plan_subgraph,  # Format: [source, dict, workload]
            workload=[projection]  # Single projection as workload
        )
        
        # Call generate_prePP using adapter
        prepp_results = run_prepp(
            input_buffer=eval_plan_buffer,
            method="ppmuse",
            algorithm="e",  # exact
            samples=0,
            top_k=0,
            runs=1,
            plan_print=True,
            all_pairs=subgraph['all_pairs_sub']
        )
        
        # Extract costs from prePP results
        if prepp_results and len(prepp_results) > 0:
            costs = prepp_results[0]  # exact_cost
            logger.info(f"PrePP costs calculated: {costs:.2f}")
            
            if all_push_baseline:
                savings = all_push_baseline - costs
                logger.info(f"Savings vs all-push: {savings:.2f} ({(savings/all_push_baseline*100):.1f}%)")
            
            return costs
        else:
            logger.warning("No valid prePP results returned, using fallback")
            return calculate_fallback_costs(node, subgraph, projection, all_push_baseline)

    except Exception as e:
        logger.error(f"Error calculating prePP costs: {e}")
        # Fallback to simple cost calculation
        return calculate_fallback_costs(node, subgraph, projection, all_push_baseline)


def _create_basic_evaluation_plan(self, subgraph: Dict[str, Any], projection: Any) -> Any:
    """Create a basic evaluation plan structure for the subgraph."""
    from EvaluationPlan import EvaluationPlan
    
    # Create empty evaluation plan
    evaluation_plan = EvaluationPlan([], [])
    
    # Initialize instances for primitive events in the subgraph
    evaluation_plan.initInstances(subgraph['index_event_nodes_sub'])
    
    return evaluation_plan


def _create_basic_central_plan(self, subgraph: Dict[str, Any], central_eval_plan: Any) -> List[Any]:
    """Create a basic central plan structure for the subgraph."""
    # Simplified central plan - use first node as source
    source_node = 0 if subgraph['sub_network'] else 0
    return [source_node, {}, []]