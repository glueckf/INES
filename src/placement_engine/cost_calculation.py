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
                                      central_eval_plan: Any, routing_algo: Any, all_push_baseline: Optional[float] = 6278.0,
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
