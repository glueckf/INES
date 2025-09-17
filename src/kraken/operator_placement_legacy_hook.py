import uuid

from helper.placement_aug import (
    NEWcomputeCentralCosts,
    ComputeSingleSinkPlacement,
    computeMSplacementCosts,
)
from helper.processCombination_aug import compute_dependencies, getSharedMSinput
import time
from .core import compute_kraken_for_projection
from .global_placement_tracker import (
    get_global_placement_tracker,
    reset_global_placement_tracker,
)
from .logging import get_kraken_logger
import csv
import os
from typing import Dict, Any

logger = get_kraken_logger(__name__)


def format_results_for_comparison(
    results_dict: Dict, execution_info: Dict, workload: list
) -> Dict[str, Any]:
    """
    Format placement results in a clean, machine-readable format for comparison.

    Args:
        results_dict: Dictionary containing placement results for each projection
        execution_info: Dictionary containing execution metadata

    Returns:
        Dictionary with structured results for easy comparison
    """
    formatted_results = {
        "metadata": execution_info,
        "placements": {},
        "summary": {
            "total_projections": len(results_dict),
            "total_cost": 0,
            "successful_placements": 0,
            "failed_placements": 0,
        },
    }

    for projection, result in results_dict.items():
        projection_str = str(projection)

        if hasattr(result, "costs") and hasattr(result, "node"):
            # New placement engine result format
            formatted_results["placements"][projection_str] = {
                "placement_node": result.node,
                "total_cost": result.costs,
                "strategy": getattr(result, "strategy", "unknown"),
                "all_push_cost": getattr(result, "all_push_costs", None),
                "push_pull_cost": getattr(result, "push_pull_costs", None),
                "has_sufficient_resources": getattr(
                    result, "has_sufficient_resources", None
                ),
                "plan_details": getattr(result, "plan_details", {}),
                "success": True,
            }
            if projection in workload:
                formatted_results["summary"]["total_cost"] += result.costs
            formatted_results["summary"]["successful_placements"] += 1
        else:
            # Handle other result formats or errors
            formatted_results["placements"][projection_str] = {
                "success": False,
                "result_type": str(type(result)),
                "result_data": str(result) if result else None,
            }
            formatted_results["summary"]["failed_placements"] += 1

    return formatted_results


def calculate_integrated_approach(self, file_path: str, max_parents: int):
    workload = self.query_workload
    projFilterDict = self.h_projFilterDict
    IndexEventNodes = self.h_IndexEventNodes
    allPairs = self.allPairs
    rates = self.h_rates_data
    network = self.network
    mycombi = self.h_mycombi
    projrates = self.h_projrates
    EventNodes = self.h_eventNodes
    G = self.graph
    selectivities = self.selectivities
    mode = self.config.mode
    EventNodes = self.h_eventNodes
    IndexEventNodes = self.h_IndexEventNodes
    unfolded = self.h_mycombi
    criticalMSTypes = self.h_criticalMSTypes
    network_data = self.h_network_data
    Filters = []

    noFilter = 0  # NO FILTER

    # Access the arguments
    filename = file_path
    number_parents = max_parents

    central_computation_result = NEWcomputeCentralCosts(
        workload, IndexEventNodes, allPairs, rates, EventNodes, self.graph
    )
    (
        central_computation_cost,
        central_computation_node,
        central_computation_longest_path,
        central_computation_routing_dict,
    ) = central_computation_result
    centralHopLatency = max(allPairs[central_computation_node])
    numberHops = sum(allPairs[central_computation_node])
    MSPlacements = {}
    start_time = time.time()

    # Initialize global placement tracker for this placement session
    reset_global_placement_tracker()  # Start fresh for each placement calculation
    global_placement_tracker = get_global_placement_tracker()

    # Initialize global event placement tracker with network data
    from .node_tracker import initialize_global_event_tracker

    initialize_global_event_tracker(h_network_data=self.h_network_data)

    hopLatency = {}

    sharedDict = getSharedMSinput(self, unfolded, projFilterDict)
    dependencies = compute_dependencies(self, unfolded, criticalMSTypes)
    processingOrder = sorted(dependencies.keys(), key=lambda x: dependencies[x])
    costs = 0

    integrated_placement_decision_by_projection = {}

    """ 
    NOTE from FINN GLÜCK 10.09.2025: 
    For some reason in some edge cases projections with really high output rates reach the top of the processing order. 
    We do not want them in our workload, at least for single node queries, that's why we filter them out here. 
    The heuristic for the filtering is: 
    
    Sum of input rates < projection output rate 
    
    Example: 
    
    SEQ(A, B) with 'AB' = 0.5, R(A) = 1000, R(B) = 5
    
    sum of input rates = R(A) + R(B) = 1005
    
    projection output rate = R(A) * R(B) * 'AB' = 1000 * 5 * 0.5 = 2500
    
    1005 < 2500 -> filter out
    """

    """ 
    NOTE FROM Finn Glück 12.09.2025: 
    Actually no subprojections with to high output rates reach the processing order. 
    Those high rate projections are always part of the workload and can therefore not be filtered out here. 
    However the problem for INES remains, 
    as these projections still get placed inside the network and not the cloud where they should be placed.
    """
    # result = update_processing_order_with_heuristics(
    #     query_workload=workload,
    #     combinations=mycombi,
    #     processing_order=processingOrder,
    #     proj_filter_dict=projFilterDict,
    #     rates=rates,
    #     projection_rates=projrates,
    # )
    #
    # (processingOrder, projrates, projFilterDict, mycombi) = result

    # TODO: This flag currently leaves out MS placement for the integrated approach, as it is not yet implemented
    #  This should be removed and the line in 184 should be commented in once MS placement is implemented
    partType = False

    for projection in (
        processingOrder
    ):  # parallelize computation for all projections at the same level
        if set(unfolded[projection]) == set(
            projection.leafs()
        ):  # initialize hop latency with maximum of children
            hopLatency[projection] = 0
        else:
            hopLatency[projection] = max(
                [hopLatency[x] for x in unfolded[projection] if x in hopLatency.keys()]
            )

        # TODO: This should be commented in, once MS placement is implemented for the integrated approach
        # partType,_,_ = returnPartitioning(self, projection, unfolded[projection], projrates ,criticalMSTypes)

        if partType:
            MSPlacements[projection] = partType

            result = computeMSplacementCosts(
                self,
                projection,
                unfolded[projection],
                partType,
                sharedDict,
                noFilter,
                G,
            )

            additional = result[0]

            costs += additional

            hopLatency[projection] += result[1]

            Filters += result[4]

            if projection.get_original(workload) in workload and partType[0] in list(
                map(
                    lambda x: str(x),
                    projection.get_original(workload).kleene_components(),
                )
            ):
                result = ComputeSingleSinkPlacement(
                    projection.get_original(workload), [projection], noFilter
                )
                additional = result[0]
                costs += additional

        else:
            integrated_optimization_result_for_given_projection = (
                compute_kraken_for_projection(
                    workload,
                    selectivities,
                    mycombi,
                    mode,
                    projection,
                    unfolded[projection],
                    noFilter,
                    projFilterDict,
                    EventNodes,
                    IndexEventNodes,
                    network_data,
                    allPairs,
                    mycombi,
                    rates,
                    projrates,
                    G,
                    network,
                )
            )

            integrated_placement_decision_by_projection[projection] = (
                integrated_optimization_result_for_given_projection
            )

    integrated_placement_decision_by_projection = finalize_placement_results(
        self=self,
        placement_decisions_by_projection=integrated_placement_decision_by_projection,
    )

    costs_for_evaluation_total_workload = 0
    # Go through each placement decision
    for projection in integrated_placement_decision_by_projection:
        costs_for_evaluation_total_workload += (
            integrated_placement_decision_by_projection[projection].costs
        )

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
                latency += integrated_placement_decision_by_projection[
                    projection
                ].plan_details.get("latency", 0)

            if latency > max_latency:
                max_latency = latency

    kraken_simulation_id = uuid.uuid4()

    end_time = time.time()
    totaltime = str(end_time - start_time)[:6]

    # Prepare execution metadata
    execution_info = {
        "experiment_id": kraken_simulation_id,
        "file_path": filename,
        "max_parents": number_parents,
        "execution_time_seconds": float(totaltime),
        "start_time": start_time,
        "end_time": end_time,
        "total_execution_time_seconds": end_time - start_time,
        "push_pull_plan_cost_sum": costs_for_evaluation_total_workload,
        "push_pull_plan_latency": max_latency,
        "central_cost": central_computation_cost,
        "central_hop_latency": centralHopLatency,
        "number_hops": numberHops,
        "workload_size": len(workload),
        "global_tracker_entries": len(global_placement_tracker._placement_history)
        if global_placement_tracker
        else 0,
    }

    # Format results for comparison
    formatted_results = format_results_for_comparison(
        integrated_placement_decision_by_projection, execution_info, workload
    )

    result = {
        "kraken_simulation_id": kraken_simulation_id,
        "integrated_placement_decision_by_projection": integrated_placement_decision_by_projection,
        "formatted_results": formatted_results,
    }

    return result


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
            return 0.0

        visited.add(projection_str)

        # Get the placement decision for this projection
        if projection not in placement_decisions_by_projection:
            return 0.0

        decision = placement_decisions_by_projection[projection]
        base_cost = decision.costs
        total_cost = base_cost

        # Check if this projection has acquisition steps
        acquisition_steps = decision.plan_details.get("aquisition_steps", {})

        if not acquisition_steps:
            return total_cost

        # Go through each acquisition step
        for step_idx, step_details in acquisition_steps.items():
            events_to_pull = step_details.get("events_to_pull", [])

            # Check each event being pulled in this step
            for event in events_to_pull:
                # Check if this event is a subprojection (not a primitive event)
                if event in projection_str_to_obj:
                    subprojection = projection_str_to_obj[event]

                    # Recursively calculate the cost of the subprojection
                    subprojection_cost = calculate_total_cost_recursive(
                        subprojection, visited.copy()
                    )
                    total_cost += subprojection_cost

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
            # Create a new decision object with updated cost
            class UpdatedDecision:
                def __init__(self, original_decision, new_total_cost):
                    # Copy all attributes from original decision
                    self.node = original_decision.node
                    self.strategy = original_decision.strategy
                    self.all_push_costs = original_decision.all_push_costs
                    self.push_pull_costs = original_decision.push_pull_costs
                    self.has_sufficient_resources = (
                        original_decision.has_sufficient_resources
                    )
                    self.plan_details = original_decision.plan_details
                    self.savings = original_decision.savings

                    # Update the total cost with aggregated cost
                    self.costs = new_total_cost
                    self.original_costs = original_decision.costs
                    self.aggregated_additional_cost = (
                        new_total_cost - original_decision.costs
                    )

            finalized_decision = UpdatedDecision(decision, total_cost)

        finalized_decisions[projection] = finalized_decision

    return finalized_decisions


def write_final_results(
    integrated_operator_placement_results, ines_results, config, graph_density
):
    print("Hook")

    # Save results to csv file for analysis, but merged with INES results
    columns_to_copy = [
        # ID
        "ines_simulation_id",
        "kraken_simulation_id",
        # Configuration parameters
        "network_size",
        "event_skew",
        "node_event_ratio",
        "num_event_types",
        "max_parents",
        "workload_size",
        "query_length",
        "simulation_mode",
        "median_selectivity",
        # Metadata
        "total_projections_placed",
        "placement_difference_to_ines_count",
        "placements_at_cloud",
        "graph_density",
        # Computation times
        "combigen_time_seconds",
        "ines_placement_time_seconds",
        "ines_push_pull_time_seconds",
        "ines_total_time_seconds",
        "kraken_execution_time_seconds",
        # Placement costs
        "all_push_central_cost",
        "inev_cost",
        "ines_cost",
        "kraken_cost",
        # Latency
        "all_push_central_latency",
        "ines_latency",
        "kraken_latency",
    ]

    kraken_metadata = integrated_operator_placement_results["formatted_results"][
        "metadata"
    ]
    kraken_summary = integrated_operator_placement_results["formatted_results"][
        "summary"
    ]

    # Extract IDs
    ines_simulation_id = ines_results[0]
    kraken_simulation_id = integrated_operator_placement_results["kraken_simulation_id"]

    # Extract config parameters
    network_size = config.network_size
    event_skew = config.event_skew
    node_event_ratio = config.node_event_ratio
    num_event_types = config.num_event_types
    max_parents = config.max_parents
    workload_size = config.query_size
    query_length = config.query_length
    simulation_mode = config.mode.value
    median_selectivity = ines_results[11]

    # Extract metadata
    total_projections_placed = kraken_summary.get("successful_placements", 0)
    placement_difference_to_ines_count = kraken_summary.get(
        "placement_difference_count", 0
    )
    placements_at_cloud = calculate_placements_at_cloud(
        integrated_operator_placement_results
    )
    graph_density = graph_density

    # Computation times
    combigen_time_seconds = ines_results[12]
    ines_placement_time_seconds = ines_results[14]
    ines_push_pull_time_seconds = ines_results[22]
    ines_total_time_seconds = float(ines_placement_time_seconds) + float(
        ines_push_pull_time_seconds
    )
    kraken_execution_time_seconds = kraken_metadata.get("execution_time_seconds", 0)

    # Placement costs
    all_push_central_cost = ines_results[2]
    inev_cost = ines_results[3]
    ines_cost = ines_results[21]
    kraken_cost = kraken_metadata.get("push_pull_plan_cost_sum", 0)

    # Latency
    all_push_central_latency = ines_results[15]
    ines_latency = ines_results[23]
    kraken_latency = kraken_metadata.get("push_pull_plan_latency", 0)

    # Compile all data into a single row
    row_data = {
        # IDs
        "ines_simulation_id": ines_simulation_id,
        "kraken_simulation_id": kraken_simulation_id,
        # Configuration parameters
        "network_size": network_size,
        "event_skew": event_skew,
        "node_event_ratio": node_event_ratio,
        "num_event_types": num_event_types,
        "max_parents": max_parents,
        "workload_size": workload_size,
        "query_length": query_length,
        "simulation_mode": simulation_mode,
        "median_selectivity": median_selectivity,
        # Metadata
        "total_projections_placed": total_projections_placed,
        "placement_difference_to_ines_count": placement_difference_to_ines_count,
        "placements_at_cloud": placements_at_cloud,
        "graph_density": graph_density,
        # Computation times
        "combigen_time_seconds": combigen_time_seconds,
        "ines_placement_time_seconds": ines_placement_time_seconds,
        "ines_push_pull_time_seconds": ines_push_pull_time_seconds,
        "ines_total_time_seconds": ines_total_time_seconds,
        "kraken_execution_time_seconds": kraken_execution_time_seconds,
        # Placement costs
        "all_push_central_cost": all_push_central_cost,
        "inev_cost": inev_cost,
        "ines_cost": ines_cost,
        "kraken_cost": kraken_cost,
        # Latency
        "all_push_central_latency": all_push_central_latency,
        "ines_latency": ines_latency,
        "kraken_latency": kraken_latency,
    }

    # Ensure result directory exists
    result_dir = os.path.join(os.path.dirname(__file__), "result")
    os.makedirs(result_dir, exist_ok=True)

    csv_file_path = os.path.join(result_dir, "run_results.csv")

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file_path)

    # Write to CSV file (append mode)
    with open(csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write headers only if file is new
        if not file_exists:
            writer.writerow(columns_to_copy)

        # Write data row using the order defined in columns_to_copy
        row_values = [row_data[col] for col in columns_to_copy]
        writer.writerow(row_values)

    logger.info(f"Appended combined simulation results to {csv_file_path}")
    logger.info(f"INES ID: {ines_simulation_id}, Kraken ID: {kraken_simulation_id}")


def calculate_placements_at_cloud(integrated_operator_placement_results):
    """
    Calculate the number of projections placed at the cloud node (node 0).

    Args:
        integrated_operator_placement_results: Complete results dict containing all placement data

    Returns:
        int: Count of projections placed at the cloud node
    """
    # Loop through each projection's placement decision and increment count if placed at cloud (node 0)
    placed_at_cloud = 0
    placement_decisions = integrated_operator_placement_results.get(
        "integrated_placement_decision_by_projection", {}
    )
    for projection, decision in placement_decisions.items():
        if hasattr(decision, "node") and decision.node == 0:
            placed_at_cloud += 1

    return placed_at_cloud
