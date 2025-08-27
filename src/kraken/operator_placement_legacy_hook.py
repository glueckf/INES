from helper.placement_aug import NEWcomputeCentralCosts, ComputeSingleSinkPlacement, computeMSplacementCosts
from helper.processCombination_aug import compute_dependencies, getSharedMSinput
import time
import numpy as np
from .core import compute_operator_placement_with_prepp
from .global_placement_tracker import get_global_placement_tracker, reset_global_placement_tracker
from .node_tracker import initialize_global_event_tracker
import json
import csv
import os
from typing import Dict, Any


def format_results_for_comparison(results_dict: Dict, execution_info: Dict, workload: list) -> Dict[str, Any]:
    """
    Format placement results in a clean, machine-readable format for comparison.
    
    Args:
        results_dict: Dictionary containing placement results for each projection
        execution_info: Dictionary containing execution metadata
        
    Returns:
        Dictionary with structured results for easy comparison
    """
    formatted_results = {
        'metadata': execution_info,
        'placements': {},
        'summary': {
            'total_projections': len(results_dict),
            'total_cost': 0,
            'successful_placements': 0,
            'failed_placements': 0
        }
    }
    
    for projection, result in results_dict.items():
        projection_str = str(projection)
        
        if hasattr(result, 'costs') and hasattr(result, 'node'):
            # New placement engine result format
            formatted_results['placements'][projection_str] = {
                'placement_node': result.node,
                'total_cost': result.costs,
                'strategy': getattr(result, 'strategy', 'unknown'),
                'all_push_cost': getattr(result, 'all_push_costs', None),
                'push_pull_cost': getattr(result, 'push_pull_costs', None),
                'has_sufficient_resources': getattr(result, 'has_sufficient_resources', None),
                'plan_details': getattr(result, 'plan_details', {}),
                'success': True
            }
            if projection in workload:
                formatted_results['summary']['total_cost'] += result.costs
            formatted_results['summary']['successful_placements'] += 1
        else:
            # Handle other result formats or errors
            formatted_results['placements'][projection_str] = {
                'success': False,
                'result_type': str(type(result)),
                'result_data': str(result) if result else None
            }
            formatted_results['summary']['failed_placements'] += 1
    
    return formatted_results


def save_results_to_json(results: Dict[str, Any], filename: str) -> None:
    """
    Save formatted results to a JSON file.
    
    Args:
        results: Formatted results dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[RESULTS] Saved machine-readable results to {filename}")


def calculate_integrated_approach(self, file_path: str, max_parents: int):
    workload = self.query_workload
    projFilterDict = self.h_projFilterDict
    IndexEventNodes = self.h_IndexEventNodes
    allPairs = self.allPairs
    rates = self.h_rates_data
    network = self.network
    mycombi = self.h_mycombi
    singleSelectivities = self.single_selectivity
    projrates = self.h_projrates
    EventNodes = self.h_eventNodes
    G = self.graph

    Filters = []

    noFilter = 0  # NO FILTER

    # Access the arguments
    filename = file_path
    number_parents = max_parents

    # print(f"[PLACEMENT] Processing file: {filename}")
    # print(f"[PLACEMENT] Index Event Nodes: {IndexEventNodes}")
    central_computation_result = NEWcomputeCentralCosts(workload, IndexEventNodes, allPairs, rates, EventNodes, self.graph)
    (central_computation_cost,
     central_computation_node,
     central_computation_longest_path,
     central_computation_routing_dict) = central_computation_result
    centralHopLatency = max(allPairs[central_computation_node])
    numberHops = sum(allPairs[central_computation_node])
    # print(f"[CENTRAL COSTS] Cost: {central_computation_cost:.2f}")
    # print(f"[CENTRAL COSTS] Total Hops: {numberHops}")
    # print(f"[CENTRAL COSTS] Hop Latency: {centralHopLatency:.2f}")
    MSPlacements = {}
    start_time = time.time()

    # Initialize global placement tracker for this placement session
    reset_global_placement_tracker()  # Start fresh for each placement calculation
    global_placement_tracker = get_global_placement_tracker()
    global_event_tracker = initialize_global_event_tracker(h_network_data=self.h_network_data)
    # print(f"[PLACEMENT] Initialized global placement tracker")

    hopLatency = {}

    EventNodes = self.h_eventNodes
    IndexEventNodes = self.h_IndexEventNodes

    unfolded = self.h_mycombi
    criticalMSTypes = self.h_criticalMSTypes
    sharedDict = getSharedMSinput(self, unfolded, projFilterDict)
    dependencies = compute_dependencies(self, unfolded, criticalMSTypes)
    processingOrder = sorted(dependencies.keys(), key=lambda x: dependencies[x])
    costs = 0

    central_evaluation_plan = [central_computation_node, central_computation_routing_dict, workload]

    integrated_placement_decision_by_projection = {}

    # processingOrder = dependency_sort(processingOrder)

    for projection in processingOrder:  #parallelize computation for all projections at the same level
        if set(unfolded[projection]) == set(projection.leafs()):  #initialize hop latency with maximum of children
            hopLatency[projection] = 0
        else:
            hopLatency[projection] = max([hopLatency[x] for x in unfolded[projection] if x in hopLatency.keys()])

        # TODO: Currntly leave out MS placement for integrated approach, as it is not yet implemented
        # partType,_,_ = returnPartitioning(self, projection, unfolded[projection], projrates ,criticalMSTypes)
        partType = False

        if partType:

            # TODO: Should be rewritten to fit the integrated approach
            MSPlacements[projection] = partType

            result = computeMSplacementCosts(self, projection, unfolded[projection], partType, sharedDict, noFilter, G)

            additional = result[0]

            costs += additional

            hopLatency[projection] += result[1]

            Filters += result[4]

            if (
                    projection.get_original(workload) in workload and
                    partType[0] in list(map(lambda x: str(x), projection.get_original(workload).kleene_components()))
            ):
                result = ComputeSingleSinkPlacement(projection.get_original(workload), [projection], noFilter)
                additional = result[0]
                costs += additional

        else:

            integrated_optimization_result_for_given_projection = compute_operator_placement_with_prepp(
                self,
                projection,
                unfolded[projection],
                noFilter,
                projFilterDict,
                EventNodes,
                IndexEventNodes,
                self.h_network_data,
                allPairs, mycombi,
                rates,
                singleSelectivities,
                projrates,
                G,
                network,
                central_evaluation_plan)

            integrated_placement_decision_by_projection[projection] = (
                integrated_optimization_result_for_given_projection)


    integrated_placement_decision_by_projection = finalize_placement_results(self, integrated_placement_decision_by_projection)

    costs_for_evaluation_total_workload = 0
    # Go through each placement decision
    for projection in integrated_placement_decision_by_projection:
        # Check if projection was in original query workload
        if projection in self.query_workload:
            # If so, add costs to the total costs
            costs_for_evaluation_total_workload += integrated_placement_decision_by_projection[projection].costs


    max_latency = 0

    for projection in integrated_placement_decision_by_projection:
        # Check if projection was in original query workload
        if projection in self.query_workload:
            relevant_projection_set = [projection]
            unfolded_projection = unfolded[projection]
            set_unfolded = set(unfolded_projection)

            for x in set_unfolded:
                if x in processingOrder:
                    relevant_projection_set.append(x)
            latency = 0
            for projection in relevant_projection_set:
                latency += integrated_placement_decision_by_projection[projection].plan_details.get('latency', 0)

            if latency > max_latency:
                max_latency = latency

    # Print global placement tracker summary
    print("\n[PLACEMENT] Global Placement Tracker Summary:")
    print(global_placement_tracker.get_summary())

    #
    kraken_simulation_id = int(np.random.uniform(0, 10000000))

    end_time = time.time()
    totaltime = str(end_time - start_time)[:6]

    print("[TIMING] Execution Summary:")
    print(f"[TIMING] Start: {start_time:.2f}")
    print(f"[TIMING] End: {time.time():.2f}")
    print(f"[TIMING] Duration: {totaltime} seconds")
    
    # Prepare execution metadata
    execution_info = {
        'experiment_id': kraken_simulation_id,
        'file_path': filename,
        'max_parents': number_parents,
        'execution_time_seconds': float(totaltime),
        'start_time': start_time,
        'end_time': end_time,
        'total_execution_time_seconds': end_time - start_time,
        'push_pull_plan_cost_sum': costs_for_evaluation_total_workload,
        'push_pull_plan_latency': max_latency,
        'central_cost': central_computation_cost,
        'central_hop_latency': centralHopLatency,
        'number_hops': numberHops,
        'workload_size': len(workload),
        'global_tracker_entries': len(global_placement_tracker._placement_history) if global_placement_tracker else 0
    }
    
    # Format results for comparison
    formatted_results = format_results_for_comparison(integrated_placement_decision_by_projection, execution_info, workload)
    
    # Print summary for immediate feedback
    print("\n[RESULTS SUMMARY]")
    print(f"Total Projections: {formatted_results['summary']['total_projections']}")
    print(f"Successful Placements: {formatted_results['summary']['successful_placements']}")
    print(f"Failed Placements: {formatted_results['summary']['failed_placements']}")
    print(f"Total Placement Cost: {formatted_results['summary']['total_cost']:.2f}")

    result = {
        'kraken_simulation_id': kraken_simulation_id,
        'integrated_placement_decision_by_projection': integrated_placement_decision_by_projection,
        'formatted_results': formatted_results
    }
    
    return result


def write_results_to_csv(integrated_placement_decision_by_projection, ines_simulation_id):
    """
    Write aggregated placement results to CSV file for comparison analysis.
    
    Args:
        integrated_placement_decision_by_projection: Dict containing placement decisions by projection
        ines_simulation_id: Foreign key ID to link with INES simulation data
    """
    
    # Ensure res directory exists
    res_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'res')
    os.makedirs(res_dir, exist_ok=True)
    
    csv_file_path = os.path.join(res_dir, 'kraken_simulation.csv')

    # Define CSV headers
    headers = [
        'kraken_simulation_id',
        'ines_simulation_id', 
        'timestamp',
        'execution_time_seconds',
        'total_projections_placed',
        'central_placement_cost',
        'integrated_placement-cost',
        'central_latency',
        'integrated_placement_latency',
        'cost_reduction_ratio',
        'latency_increase',
        'workload_size',
        'nodes_with_placements'
    ]

    kraken_simulation_id = integrated_placement_decision_by_projection['kraken_simulation_id']
    timestamp = integrated_placement_decision_by_projection['formatted_results']['metadata']['start_time']
    execution_time_seconds = integrated_placement_decision_by_projection['formatted_results']['metadata']['execution_time_seconds']
    total_projections_placed = integrated_placement_decision_by_projection['formatted_results']['summary']['total_projections']
    central_placement_cost = integrated_placement_decision_by_projection['formatted_results']['metadata']['central_cost']
    integrated_placement_cost = integrated_placement_decision_by_projection['formatted_results']['metadata']['total_cost_sum']
    central_latency = integrated_placement_decision_by_projection['formatted_results']['metadata']['central_hop_latency']

    # TODO: Not returned as of right now, needs to be calculated
    integrated_placement_latency = central_latency

    cost_reduction_ratio = integrated_placement_cost / central_placement_cost
    latency_increase = (integrated_placement_latency / central_latency) - 1
    workload_size = integrated_placement_decision_by_projection['formatted_results']['metadata']['workload_size']
    nodes_with_placements = integrated_placement_decision_by_projection['formatted_results']['metadata']['global_tracker_entries']

    row_data = [
        kraken_simulation_id,
        ines_simulation_id,
        timestamp,
        execution_time_seconds,
        total_projections_placed,
        central_placement_cost,
        integrated_placement_cost,
        central_latency,
        integrated_placement_latency,
        cost_reduction_ratio,
        latency_increase,
        workload_size,
        nodes_with_placements
    ]

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file_path)

    # Write to CSV file (append mode)
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write headers only if file is new
        if not file_exists:
            writer.writerow(headers)

        # Write data row
        writer.writerow(row_data)

    print(f"[CSV] Appended Kraken simulation results to {csv_file_path}")
    print(f"[CSV] Kraken Simulation ID: {kraken_simulation_id}, INES Simulation ID: {ines_simulation_id}")


def print_kraken(integrated_operator_placement_results):
    """
    Print comprehensive placement results information for Kraken simulation.
    
    Args:
        integrated_operator_placement_results: Complete results dict containing all placement data
    """
    print("\n" + "="*80)
    print("KRAKEN SIMULATION RESULTS SUMMARY")
    print("="*80)
    
    # Extract main components
    kraken_id = integrated_operator_placement_results.get('kraken_simulation_id', 'N/A')
    placement_decisions = integrated_operator_placement_results.get('integrated_placement_decision_by_projection', {})
    formatted_results = integrated_operator_placement_results.get('formatted_results', {})
    
    # Print simulation metadata
    metadata = formatted_results.get('metadata', {})
    print("\nSIMULATION METADATA:")
    print(f"   Kraken Simulation ID: {kraken_id}")
    print(f"   Experiment ID: {metadata.get('experiment_id', 'N/A')}")
    print(f"   File Path: {metadata.get('file_path', 'N/A')}")
    print(f"   Execution Time: {metadata.get('execution_time_seconds', 0):.2f} seconds")
    print(f"   Workload Size: {metadata.get('workload_size', 0)} queries")
    print(f"   Max Parents: {metadata.get('max_parents', 0)}")
    
    # Print cost comparison
    central_cost = metadata.get('central_cost', 0)
    total_integrated_cost = metadata.get('total_cost_sum', 0)
    savings = central_cost - total_integrated_cost if central_cost > 0 else 0
    savings_pct = (savings / central_cost * 100) if central_cost > 0 else 0
    
    print("\nCOST ANALYSIS:")
    print(f"   Central Placement Cost: {central_cost:,.2f}")
    print(f"   Integrated Placement Cost: {total_integrated_cost:,.2f}")
    print(f"   Total Savings: {savings:,.2f} ({savings_pct:.1f}%)")
    
    # Print latency information
    print("\nLATENCY METRICS:")
    print(f"   Central Hop Latency: {metadata.get('central_hop_latency', 0)} hops")
    print(f"   Total Network Hops: {metadata.get('number_hops', 0)}")
    
    # Print placement summary
    summary = formatted_results.get('summary', {})
    print("\nPLACEMENT SUMMARY:")
    print(f"   Total Projections: {summary.get('total_projections', 0)}")
    print(f"   Successful Placements: {summary.get('successful_placements', 0)}")
    print(f"   Failed Placements: {summary.get('failed_placements', 0)}")
    print(f"   Success Rate: {(summary.get('successful_placements', 0) / max(summary.get('total_projections', 1), 1) * 100):.1f}%")
    
    # Print detailed placement decisions
    print("\nDETAILED PLACEMENT DECISIONS:")
    for projection, decision in placement_decisions.items():
        if hasattr(decision, 'node') and hasattr(decision, 'costs'):
            print(f"   {projection}:")
            print(f"      Node: {decision.node}")
            print(f"      Strategy: {decision.strategy}")
            print(f"      Final Cost: {decision.costs:,.2f}")
            print(f"      All-Push Cost: {decision.all_push_costs:,.2f}")
            print(f"      Savings: {decision.savings:,.2f}")
            print(f"      Resources Sufficient: {decision.has_sufficient_resources}")
            
            # Print plan details if available
            if hasattr(decision, 'plan_details') and decision.plan_details:
                details = decision.plan_details
                print(f"      Computing Time: {details.get('computing_time', 0):.4f}s")
                print(f"      Latency: {details.get('latency', 0)} hops")
                print(f"      Transmission Ratio: {details.get('transmission_ratio', 0):.6f}")
            print()
    
    # Print formatted placements summary
    placements = formatted_results.get('placements', {})
    if placements:
        print("PLACEMENT NODES DISTRIBUTION:")
        node_counts = {}
        for proj_name, placement_info in placements.items():
            node = placement_info.get('placement_node', 'unknown')
            node_counts[node] = node_counts.get(node, 0) + 1
        
        for node, count in sorted(node_counts.items()):
            print(f"   Node {node}: {count} projection(s)")
    
    print("="*80)
    print(f"Kraken simulation {kraken_id} completed successfully!")
    print("="*80)


def finalize_placement_results(self, placement_decisions_by_projection):
    """
    Aggregate placement costs by recursively following acquisition steps and adding subprojection costs.
    
    For each workload query, we need to find all subprojections that were acquired in the acquisition steps
    and add their placement costs to get the total cost for placing the workload query.
    
    Args:
        self: The simulation context containing workload and unfolded projections
        placement_decisions_by_projection: Dict mapping projections to their PlacementDecision objects
        
    Returns:
        Dict: Updated placement decisions with aggregated costs for workload queries
    """
    print("[FINALIZE] Starting placement result finalization")
    workload = self.query_workload
    unfolded_workload = self.h_mycombi
    
    print(f"[FINALIZE] Workload queries: {[str(q) for q in workload]}")
    print(f"[FINALIZE] Total projections placed: {len(placement_decisions_by_projection)}")
    
    # Create a mapping from string representation to projection objects for lookup
    projection_str_to_obj = {}
    for proj in placement_decisions_by_projection.keys():
        projection_str_to_obj[str(proj)] = proj
    
    def calculate_total_cost_recursive(projection, visited=None):
        """
        Recursively calculate the total cost for a projection including all subprojection costs.
        
        Args:
            projection: The projection object to calculate costs for
            visited: Set to track visited projections to avoid infinite recursion
            
        Returns:
            float: Total aggregated cost including all subprojections
        """
        if visited is None:
            visited = set()
            
        projection_str = str(projection)
        
        # Avoid infinite recursion
        if projection_str in visited:
            print(f"[FINALIZE] Cycle detected for {projection_str}, skipping")
            return 0.0
            
        visited.add(projection_str)
        
        # Get the placement decision for this projection
        if projection not in placement_decisions_by_projection:
            print(f"[FINALIZE] No placement decision found for {projection_str}")
            return 0.0
            
        decision = placement_decisions_by_projection[projection]
        base_cost = decision.costs
        total_cost = base_cost
        
        print(f"[FINALIZE] Processing {projection_str} (base cost: {base_cost:.2f})")
        
        # Check if this projection has acquisition steps
        acquisition_steps = decision.plan_details.get('aquisition_steps', {})
        
        if not acquisition_steps:
            print(f"[FINALIZE] No acquisition steps for {projection_str}")
            return total_cost
            
        # Go through each acquisition step
        for step_idx, step_details in acquisition_steps.items():
            events_to_pull = step_details.get('events_to_pull', [])
            
            print(f"[FINALIZE] Step {step_idx} events_to_pull: {events_to_pull}")
            
            # Check each event being pulled in this step
            for event in events_to_pull:
                # Check if this event is a subprojection (not a primitive event)
                if event in projection_str_to_obj:
                    subprojection = projection_str_to_obj[event]
                    print(f"[FINALIZE] Found subprojection: {event}")
                    
                    # Recursively calculate the cost of the subprojection
                    subprojection_cost = calculate_total_cost_recursive(subprojection, visited.copy())
                    total_cost += subprojection_cost
                    
                    print(f"[FINALIZE] Added subprojection cost {subprojection_cost:.2f} for {event}")
                else:
                    # This is a primitive event, no additional cost
                    print(f"[FINALIZE] Primitive event: {event}")
        
        print(f"[FINALIZE] Total aggregated cost for {projection_str}: {total_cost:.2f}")
        return total_cost
    
    # Create updated placement decisions with aggregated costs
    finalized_decisions = {}
    
    # Process each projection
    for projection, decision in placement_decisions_by_projection.items():
        # Calculate the total aggregated cost
        total_cost = calculate_total_cost_recursive(projection)
        
        # Create a new decision object with updated costs
        # We'll create a copy of the original decision and update the costs
        finalized_decision = decision  # Start with original decision
        
        # Update the costs if aggregation found additional costs
        if total_cost != decision.costs:
            print(f"[FINALIZE] Updating costs for {str(projection)}: {decision.costs:.2f} -> {total_cost:.2f}")
            
            # Create a new decision object with updated cost
            class UpdatedDecision:
                def __init__(self, original_decision, new_total_cost):
                    # Copy all attributes from original decision
                    self.node = original_decision.node
                    self.strategy = original_decision.strategy
                    self.all_push_costs = original_decision.all_push_costs
                    self.push_pull_costs = original_decision.push_pull_costs
                    self.has_sufficient_resources = original_decision.has_sufficient_resources
                    self.plan_details = original_decision.plan_details
                    self.savings = original_decision.savings
                    
                    # Update the total cost with aggregated cost
                    self.costs = new_total_cost
                    self.original_costs = original_decision.costs
                    self.aggregated_additional_cost = new_total_cost - original_decision.costs
            
            finalized_decision = UpdatedDecision(decision, total_cost)
        
        finalized_decisions[projection] = finalized_decision
    
    # Print summary of finalization
    print(f"\n[FINALIZE] Summary:")
    total_original_cost = sum(d.costs if hasattr(d, 'original_costs') else d.costs 
                            for d in placement_decisions_by_projection.values() 
                            if hasattr(d, 'costs'))
    total_finalized_cost = sum(d.costs for d in finalized_decisions.values() if hasattr(d, 'costs') and d in workload)
    
    print(f"[FINALIZE] Original total cost: {total_original_cost:.2f}")
    print(f"[FINALIZE] Finalized total cost: {total_finalized_cost:.2f}")
    print(f"[FINALIZE] Additional aggregated cost: {total_finalized_cost - total_original_cost:.2f}")
    
    return finalized_decisions
