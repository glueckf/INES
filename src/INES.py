import string
from enum import Enum
from dataclasses import dataclass
from typing import Dict

import networkx as nx
import numpy as np

from Node import Node
from network import generate_eventrates, create_random_tree
from graph import create_fog_graph
from allPairs import populate_allPairs
from queryworkload import generate_workload
from selectivity import initialize_selectivities
from write_config_single import generate_config_buffer
from singleSelectivities import initializeSingleSelectivity
from helper.parse_network import initialize_globals
from helper.structures import initEventNodes
from combigen import populate_projFilterDict, removeFilters, generate_combigen
from helper.structures import getLongest
from operatorplacement import calculate_operatorPlacement


class SimulationMode(Enum):
    """Defines different simulation configuration modes for reproducible experiments."""

    RANDOM = "random"  # All components randomly generated
    FIXED_TOPOLOGY = "fixed_topology"  # Fixed network topology, rest random
    FIXED_WORKLOAD = "fixed_workload"  # Fixed topology + workload, rest random
    FULLY_DETERMINISTIC = "deterministic"  # All components fixed for reproducibility


class PlacementAlgorithm(Enum):
    """Defines available placement algorithms for Kraken."""

    GREEDY = "greedy"  # Greedy algorithm (default, fast)
    BACKTRACKING = "backtracking"  # Backtracking with latency constraints
    BRANCH_AND_CUT = "branch_and_cut"  # Branch and cut (not yet implemented)


@dataclass
class SimulationConfig:
    """Configuration class for INES simulation parameters."""

    # Network parameters
    network_size: int = 12
    event_skew: float = 0.3
    node_event_ratio: float = 0.5
    max_parents: int = 10
    parent_factor: float = 1.8
    num_event_types: int = 6

    # Query parameters
    query_size: int = 3
    query_length: int = 5

    # Latency Awareness
    xi: float = 0.0
    latency_threshold: float = None  # If None, latency is not considered
    cost_weight: float = 0.5

    # Simulation mode
    mode: SimulationMode = SimulationMode.RANDOM

    # Placement algorithm
    algorithm: PlacementAlgorithm = PlacementAlgorithm.GREEDY

    def is_topology_fixed(self) -> bool:
        """Check if network topology should be hardcoded."""
        return self.mode in [
            SimulationMode.FIXED_TOPOLOGY,
            SimulationMode.FIXED_WORKLOAD,
            SimulationMode.FULLY_DETERMINISTIC,
        ]

    def is_workload_fixed(self) -> bool:
        """Check if query workload should be hardcoded."""
        return self.mode in [
            SimulationMode.FIXED_WORKLOAD,
            SimulationMode.FULLY_DETERMINISTIC,
        ]

    def is_selectivities_fixed(self) -> bool:
        """Check if selectivities should be hardcoded."""
        return self.mode == SimulationMode.FULLY_DETERMINISTIC

    @classmethod
    def create_random(cls, **kwargs) -> "SimulationConfig":
        """Create a fully random simulation configuration."""
        return cls(mode=SimulationMode.RANDOM, **kwargs)

    @classmethod
    def create_fixed_topology(cls, **kwargs) -> "SimulationConfig":
        """Create configuration with fixed topology, rest random."""
        return cls(mode=SimulationMode.FIXED_TOPOLOGY, **kwargs)

    @classmethod
    def create_fixed_workload(cls, **kwargs) -> "SimulationConfig":
        """Create configuration with fixed topology and workload, rest random."""
        return cls(mode=SimulationMode.FIXED_WORKLOAD, **kwargs)

    @classmethod
    def create_deterministic(cls, **kwargs) -> "SimulationConfig":
        """Create fully deterministic configuration for reproducible results."""
        return cls(mode=SimulationMode.FULLY_DETERMINISTIC, **kwargs)


from prepp import generate_prePP


def create_hardcoded_tree():
    """
    Create a hardcoded topology with 12 nodes in a hierarchical structure:
    - Layer 0: Node 0 (Cloud)
    - Layer 1: Nodes 1, 2
    - Layer 2: Nodes 3, 4, 5
    - Layer 3: Nodes 6, 7, 8, 9, 10, 11 (Leaf nodes)

    Based on expected output structure where all intermediate nodes connect to all leaf nodes.
    """
    from Node import Node
    import math

    # Initialize network and event tracking
    nw = []
    eList = {}

    # Define event rates with strong variation
    base_eventrates = generate_hardcoded_primitive_events()

    # Create nodes with decreasing compute power by layer
    nodes = {}

    # Layer 0: Cloud (Node 0)
    nodes[0] = Node(id=0, compute_power=math.inf, memory=math.inf)
    nodes[0].eventrates = [0] * len(base_eventrates)
    nw.append(nodes[0])

    # Layer 1: Nodes 1, 2
    for node_id in [1, 2]:
        nodes[node_id] = Node(id=node_id, compute_power=30, memory=30)
        nodes[node_id].eventrates = [0] * len(base_eventrates)
        nw.append(nodes[node_id])

    # Layer 2: Nodes 3, 4, 5
    for node_id in [3, 4, 5]:
        nodes[node_id] = Node(id=node_id, compute_power=20, memory=20)
        nodes[node_id].eventrates = [0] * len(base_eventrates)
        nw.append(nodes[node_id])

    # Layer 3: Leaf nodes 6, 7, 8, 9, 10, 11
    for node_id in [6, 7, 8, 9, 10, 11]:
        nodes[node_id] = Node(id=node_id, compute_power=10, memory=10)
        nodes[node_id].eventrates = [0] * len(base_eventrates)
        nw.append(nodes[node_id])

    # Set up parent-child relationships based on expected output
    # Cloud connections (Layer 0 -> Layer 1)
    nodes[0].Child = [nodes[1], nodes[2]]
    nodes[1].Parent = [nodes[0]]
    nodes[2].Parent = [nodes[0]]

    # Layer 1 -> Layer 2 connections (partial connectivity, not full mesh)
    nodes[1].Child = [nodes[3], nodes[5]]  # Node 1 connects only to nodes 3, 5
    nodes[2].Child = [nodes[3], nodes[4]]  # Node 2 connects only to nodes 3, 4

    # Each node in layer 2 has selected parents
    nodes[3].Parent = [nodes[1], nodes[2]]  # Node 3 has both parents
    nodes[4].Parent = [nodes[2]]  # Node 4 has only node 2 as parent
    nodes[5].Parent = [nodes[1]]  # Node 5 has only node 1 as parent

    # Layer 2 -> Layer 3 connections (realistic overlap with varied parent counts)
    nodes[3].Child = [
        nodes[6],
        nodes[7],
        nodes[8],
        nodes[9],
    ]  # Node 3 connects to 6,7,8,9
    nodes[4].Child = [
        nodes[6],
        nodes[7],
        nodes[8],
        nodes[9],
        nodes[10],
    ]  # Node 4 connects to 6,7,8,9,10
    nodes[5].Child = [nodes[6], nodes[10], nodes[11]]  # Node 5 connects to 6,10,11

    # Realistic parent distribution: mix of 1, 2, and 3 parents per leaf node
    nodes[6].Parent = [nodes[3], nodes[4], nodes[5]]  # Node 6: 3 parents (3,4,5)
    nodes[7].Parent = [nodes[3], nodes[4]]  # Node 7: 1 parent (3, 4)
    nodes[8].Parent = [nodes[3], nodes[4]]  # Node 8: 2 parents (3,4)
    nodes[9].Parent = [nodes[3], nodes[4]]  # Node 9: 2 parents (3,4)
    nodes[10].Parent = [nodes[4], nodes[5]]  # Node 10: 2 parents (4,5)
    nodes[11].Parent = [nodes[5]]  # Node 11: 1 parent (5)

    # Assign events to leaf nodes with varied distributions
    # Event assignments with diverse event rate combinations
    event_assignments = {
        6: [0, 4],  # Node 6: Events A (1000), E (800)
        7: [1, 2, 3],  # Node 7: Events B (2), C (45), D (203)
        # 8: [0, 2, 4],          # Node 8: Events A (1000), C (45), E (800)
        8: [0, 2],  # Node 8: Events A (1000), C (45)
        # 9: [3,5]
        9: [3],  # Node 9: Events D (203), F (5)
        10: [0, 1, 5],  # Node 10: Events A (1000), B (2), F (5)
        11: [2, 3, 4, 5],  # Node 11: Events C (45), D (203), E (800), F (5)
    }

    # Apply event assignments
    for leaf_id, event_indices in event_assignments.items():
        for event_idx in event_indices:
            nodes[leaf_id].eventrates[event_idx] = base_eventrates[event_idx]

    # Build eList - each leaf node gets its actual ancestor IDs based on the new topology
    eList[6] = [
        0,
        1,
        2,
        3,
        4,
        5,
    ]  # Node 6: ancestors through nodes 3, 4, and 5 (all paths)
    eList[7] = [0, 1, 2, 3, 4]  # Node 7: ancestors through node 3 and 4
    eList[8] = [0, 1, 2, 3, 4]  # Node 8: ancestors through nodes 3 and 4
    eList[9] = [0, 1, 2, 3, 4]  # Node 9: ancestors through nodes 3 and 4
    eList[10] = [0, 1, 2, 4, 5]  # Node 10: ancestors through nodes 4 and 5
    eList[11] = [0, 1, 5]  # Node 11: ancestors through node 5 only

    root = nodes[0]
    return root, nw, eList


def generate_hardcoded_workload():
    """
    Generate a hardcoded workload with 4 queries showing synergies:
    - 2 simple queries
    - 1 medium complexity query
    - 1 complex nested query

    Queries share common subexpressions for optimization potential.
    """
    from helper.Tree import PrimEvent, SEQ, AND
    from queryworkload import number_children

    queries = []

    # Query 1: Simple SEQ - SEQ(A, B, C)
    q1 = SEQ(PrimEvent("A"), PrimEvent("B"), PrimEvent("C"))
    q1 = number_children(q1)
    queries.append(q1)

    # Query 2:
    q2 = AND(PrimEvent("A"), PrimEvent("B"))
    q2 = number_children(q2)
    queries.append(q2)

    # Query 3: Simple AND with shared elements - AND(A, B, D)
    q3 = AND(PrimEvent("A"), PrimEvent("B"), PrimEvent("D"))
    q3 = number_children(q3)
    queries.append(q3)
    #
    # Query 4: Medium complexity - SEQ(A, B, AND(E, F))
    # Shares A, B with queries 1 and 2
    q4 = SEQ(PrimEvent("A"), PrimEvent("B"), AND(PrimEvent("E"), PrimEvent("F")))
    q4 = number_children(q4)
    queries.append(q4)
    #
    # Query 4: Complex nested - AND(SEQ(A, B, C), D, SEQ(E, F))
    # Shares SEQ(A, B, C) with query 1, and has synergy with query 3
    q5 = AND(
        SEQ(PrimEvent("A"), PrimEvent("B"), PrimEvent("C")),
        PrimEvent("D"),
        SEQ(PrimEvent("E"), PrimEvent("F")),
    )
    q5 = number_children(q5)
    queries.append(q5)

    # Query 6: AND(A, B, C)
    q6 = AND(PrimEvent("A"), PrimEvent("B"), PrimEvent("C"))
    q6 = number_children(q6)
    queries.append(q6)

    return queries


def generate_hardcoded_selectivities():
    """
    Generate hardcoded selectivities for consistent results across simulation runs.
    This eliminates the randomness in selectivity generation.

    This function generates selectivities for all combinations of events A-F,
    using the selectivities from the first run of the original random simulation.
    """
    # From the original run output - these are the selectivities that were used
    # in the first simulation run to ensure consistency
    selectivities = {
        "CB": 1,
        "BC": 1,  # C-B and B-C: 100% selectivity
        "AD": 0.0394089916562073,
        "DA": 0.0394089916562073,  # A-D and D-A: ~3.9%
        "CA": 0.050898519659186986,
        "AC": 0.050898519659186986,  # C-A and A-C: ~5.1%
        "DF": 1,
        "FD": 1,  # D-F and F-D: 100% selectivity
        "ED": 1,
        "DE": 1,  # E-D and D-E: 100% selectivity
        "EA": 0.06653100823467012,
        "AE": 0.06653100823467012,  # E-A and A-E: ~6.7%
        "DB": 0.08737603662227181,
        "BD": 0.08737603662227181,  # D-B and B-D: ~8.7%
        "EB": 0.08525918368867062,
        "BE": 0.08525918368867062,  # E-B and B-E: ~8.5%
        "CF": 0.032539286285100014,
        "FC": 0.032539286285100014,  # C-F and F-C: ~3.3%
        "EC": 0.09001062335728674,
        "CE": 0.09001062335728674,  # E-C and C-E: ~9.0%
        "CD": 1,
        "DC": 1,  # C-D and D-C: 100% selectivity
        "AB": 0.0013437,
        "BA": 0.0013437,  # A-B and B-A: ~0.13%
        "EF": 0.04513571523602232,
        "FE": 0.04513571523602232,  # E-F and F-E: ~4.5%
        "BF": 0.09888225719599508,
        "FB": 0.09888225719599508,  # B-F and F-B: ~9.9%
        "AF": 0.024810748715466777,
        "FA": 0.024810748715466777,  # A-F and F-A: ~2.5%
    }

    for key in selectivities:
        selectivities[key] /= 1

    # Generate experiment data (used for analysis)
    selectivity_values = list(selectivities.values())
    import numpy as np

    selectivitiesExperimentData = [
        0.01,
        np.median(selectivity_values),
    ]  # [min_bound, median]

    return selectivities, selectivitiesExperimentData


def generate_hardcoded_primitive_events():
    """
    Generate hardcoded primitive events for consistent results across simulation runs.
    This ensures that single_selectivity calculations remain consistent when using hardcoded selectivities.

    Returns a fixed event distribution that matches the expected selectivity patterns.
    """
    # Fixed primitive events distribution: A=1000, B=0, C=1000, D=0, E=0, F=0
    # This distribution is consistent with the hardcoded selectivities above
    # return [10000, 20, 450, 203, 800, 5]
    return [1000, 2, 45, 203, 800, 5]  # A=1000, B=2, C=45, D=203, E=800, F=5


def calculate_global_eventrates(self):
    # print("Hook")
    global_event_rates = np.zeros_like(self.network[0].eventrates)
    for node in self.network:
        if len(node.eventrates) > 0:
            # Take every eventrate from the node
            global_event_rates = np.add(global_event_rates, node.eventrates)
    result = list(global_event_rates)
    return result


def calculate_different_placement(eval_plan: list, integrated_results: dict) -> int:
    """Compare placement differences between eval plan and Kraken placement decisions.

    This function analyzes the differences between the original placement nodes from
    the evaluation plan and the placement decisions made by the Kraken placement engine.
    It counts how many projection placements differ between the two approaches.

    Args:
        eval_plan: List containing evaluation plan with projections. The first element
            contains the projections with their original placement information.
        integrated_results: Dictionary containing Kraken's integrated placement results.
            Must include 'integrated_placement_decision_by_projection' key mapping
            projection names to placement decision objects with 'node' attributes.

    Returns:
        The total number of projection node placements that differ between the
        original eval plan and Kraken's placement decisions.
    """
    # Extract projections from the first evaluation plan element
    projections = eval_plan[0].projections

    # Get Kraken's placement decisions mapped by projection name
    kraken_placements = integrated_results[
        "integrated_placement_decision_by_projection"
    ]

    # Initialize counter for placement differences
    placement_difference_count = 0

    # Compare each projection's placement nodes
    for projection in projections:
        # Get the projection identifier name
        projection_name = projection.name.name

        # Original placement nodes from the eval plan
        original_placement_nodes = projection.name.sinks

        # Kraken's placement decision for this projection (as single-item list for comparison)
        kraken_placement_nodes = [kraken_placements[projection_name].node]

        # Count nodes that are in original placement but not in Kraken's placement
        for node in original_placement_nodes:
            if node not in kraken_placement_nodes:
                placement_difference_count += 1

    return placement_difference_count


def calculate_graph_density(graph):
    """Calculate the density of our topology."""
    density = nx.density(graph)

    return density


def update_prepp_results(self, prepp_results):
    """
    Update prepp results to include cloud transmission costs and latency calculation.

    Args:
        prepp_results: Results from generate_prePP containing:
            [0] exact_cost (total costs)
            [1] pushPullTime (calculation time)
            [2] maxPushPullLatency (correct latency value)
            [3] endTransmissionRatio
            [4] total_cost
            [5] node_received_eventtypes
            [6] acquisition_steps

    Returns:
        Tuple of (updated_costs, calculation_time, latency, final_transmission_ratio)
    """
    CLOUD_NODE_ID = 0

    query_workload = self.query_workload
    prepp_eval_plan = self.eval_plan[0].projections

    total_costs = prepp_results[0]
    calculation_time = prepp_results[1]
    max_push_pull_latency_tuple = prepp_results[
        2
    ]  # This is now a tuple (node, latency) from generate_prePP
    max_push_pull_latency = (
        max_push_pull_latency_tuple[1]
        if isinstance(max_push_pull_latency_tuple, tuple)
        else max_push_pull_latency_tuple
    )  # Extract latency for backwards compatibility
    total_push_costs = prepp_results[4]
    acquisition_steps = prepp_results[6]

    # We need to add the costs of sending the events to the cloud to the prepp results

    all_sinks_from_workload = []

    for i in prepp_eval_plan:
        projection = i.name.name
        if projection in query_workload:
            # Here we need to calculate the costs for sending query from placement to sink:
            placement_nodes = i.name.sinks
            for placed_node in placement_nodes:
                all_sinks_from_workload.append(placed_node)
                hops_from_node_to_cloud = self.allPairs[placed_node][CLOUD_NODE_ID]
                query_output_rate = self.h_projrates.get(projection, 1.0)[1]
                total_costs += hops_from_node_to_cloud * query_output_rate

    final_transmission_ratio = (
        total_costs / total_push_costs if total_push_costs > 0 else 0
    )

    # Now we need to update our latency calculation to include the hops to the cloud
    # Herefore I've modified the max_push_pull_latency to be a tuple (node, latency)
    # Then node represents the node, where the highest latency was "measured"
    if max_push_pull_latency_tuple and isinstance(max_push_pull_latency_tuple, tuple):
        max_push_pull_latency += self.allPairs[max_push_pull_latency_tuple[0]][
            CLOUD_NODE_ID
        ]
    return (
        total_costs,
        calculation_time,
        max_push_pull_latency,
        final_transmission_ratio,
        acquisition_steps,
    )


class INES:
    allPairs: list
    network: list[Node]
    eventrates: list[list[int]]
    query_workload = None
    selectivities = None
    selectivitiesExperimentData = None
    primitiveEvents: list[int]
    config_single: None
    single_selectivity = None
    nwSize: int
    node_event_ratio: float
    number_eventtypes: int
    eventskew: float
    max_parents: int
    query_size: int
    query_length: int
    networkParams: list
    eval_plan = None
    central_eval_plan = None
    experiment_result = None
    prim = None
    CURRENT_SECTION = ""
    eList = {}
    h_treeDict = {}
    graph_density = None

    "Helper Variables from different Files - namespace issues"
    h_network_data = None
    h_rates_data = None
    h_primEvents = None
    h_instances = None
    h_nodes = None
    h_projlist = []
    h_projrates = {}
    h_projsPerQuery = {}
    h_sharedProjectionsDict = {}
    h_sharedProjectionsList = []
    h_eventNodes = None
    h_IndexEventNodes = None
    h_projFilterDict = None
    h_longestPath = None
    h_mycombi = None
    # h_combiDict = None
    h_criticalMSTypes_criticalMSProjs = None
    h_combiExperimentData = None
    h_criticalMSTypes = None
    h_criticalMSProjs = None
    h_combiDict = {}
    h_globalPartitioninInputTypes = {}
    h_globalSiSInputTypes = {}
    h_placementTreeDict = {}

    def __init__(self, config: SimulationConfig):
        """
        Initialize INES simulation with the provided configuration.

        Args:
            config: SimulationConfig object containing all simulation parameters
        """

        # Store configuration and extract parameters
        self.config = config
        self.nwSize = config.network_size
        self.node_event_ratio = config.node_event_ratio
        self.number_eventtypes = config.num_event_types
        self.eventskew = config.event_skew
        self.max_parents = config.max_parents
        self.query_size = config.query_size
        self.query_length = config.query_length
        self.latency_threshold = config.latency_threshold
        # Initialize result schema for experiments
        self.schema = [
            "ID",
            "TransmissionRatio",
            "Transmission",
            "INEvTransmission",
            "FilterUsed",
            "Nodes",
            "EventSkew",
            "EventNodeRatio",
            "WorkloadSize",
            "NumberProjections",
            "MinimalSelectivity",
            "MedianSelectivity",
            "CombigenComputationTime",
            "Efficiency",
            "PlacementComputationTime",
            "centralHopLatency",
            "Depth",
            "CentralTransmission",
            "LowerBound",
            "EventTypes",
            "MaximumParents",
            "exact_costs",
            "PushPullTime",
            "MaxPushPullLatency",
            "endTransmissionRatio",
        ]

        # Initialize core simulation parameters
        from projections import generate_all_projections

        eventrates_per_source = generate_eventrates(
            config.event_skew, config.num_event_types
        )
        self.eventrates = eventrates_per_source
        self.networkParams = [
            self.eventskew,
            self.number_eventtypes,
            self.node_event_ratio,
            self.nwSize,
            min(self.eventrates) / max(self.eventrates),
        ]

        # Generate primitive events - use hardcoded values for consistent selectivity calculations
        if config.is_selectivities_fixed():
            self.primitiveEvents = generate_hardcoded_primitive_events()
        else:
            self.primitiveEvents = eventrates_per_source

        # Initialize simulation components based on configuration mode
        self._initialize_network_topology()
        self._initialize_network_graph()
        self._initialize_query_workload()
        self._initialize_selectivities()

        self.graph_density = calculate_graph_density(self.graph)

        global_event_rates = calculate_global_eventrates(self)
        self.primitiveEvents = global_event_rates
        self.eventrates = global_event_rates

        # Generate configuration and single selectivities for detailed analysis
        self.config_single = generate_config_buffer(
            self.network, self.query_workload, self.selectivities
        )
        deterministic_flag = self.config.is_selectivities_fixed()

        """
        Finn Glück 31.08.2025: 
        When using multiple runs the legacy system produces an error since some selectivities are not initialized properly.
        This is a quick fix to ensure that all selectivities are initialized properly.
        """
        # Get all events
        all_events_array_string = list(string.ascii_uppercase[: config.num_event_types])

        self.single_selectivity = initializeSingleSelectivity(
            CURRENT_SECTION=self.CURRENT_SECTION,
            config_single=self.config_single,
            workload=self.query_workload,
            is_deterministic=deterministic_flag,
            all_events_array_string=all_events_array_string,
        )

        # Initialize remaining simulation components (legacy processing pipeline)
        (
            self.h_network_data,
            self.h_rates_data,
            self.h_primEvents,
            self.h_instances,
            self.h_nodes,
        ) = initialize_globals(self.network)
        self.h_eventNodes, self.h_IndexEventNodes = initEventNodes(
            self.h_nodes, self.h_network_data
        )
        (
            self.h_projlist,
            self.h_projrates,
            self.h_projsPerQuery,
            self.h_sharedProjectionsDict,
            self.h_sharedProjectionsList,
        ) = generate_all_projections(self)
        self.h_projFilterDict = populate_projFilterDict(self)
        self.h_projFilterDict = removeFilters(self)
        (
            self.h_mycombi,
            self.h_combiDict,
            self.h_criticalMSTypes_criticalMSProjs,
            self.h_combiExperimentData,
            self.h_primitive_events,
        ) = generate_combigen(self)
        self.h_criticalMSTypes, self.h_criticalMSProjs = (
            self.h_criticalMSTypes_criticalMSProjs
        )

        # Create optimized rate lookup structure for fast dependency rate calculations
        self.h_local_rate_lookup = self._create_optimized_rate_lookup()

        # Calculate sum of primitive input rates per query for processing latency calculations
        self.sum_of_input_rates_per_query = (
            self._calculate_sum_of_primitive_input_rates_per_query()
        )

        # DEBUG: Print all available data types and their structures
        # self._debug_print_all_data()

        # Call kraken2_0
        from src.kraken2_0.run import run_kraken_solver

        results = run_kraken_solver(
            ines_context=self,
            strategies_to_run=["greedy"],
            enable_detailed_logging=False,
        )

        self.kraken_results = results

        (
            self.eval_plan,
            self.central_eval_plan,
            self.experiment_result,
            self.results,
        ) = calculate_operatorPlacement(self, "test", 0)

        # Add prepp results to complete the schema (4 additional columns)
        from generateEvalPlan import generate_eval_plan

        self.plan, reused_buffer_section = generate_eval_plan(
            self.network,
            self.selectivities,
            self.eval_plan,
            self.central_eval_plan,
            self.query_workload,
        )
        self.plan_content = self.plan.getvalue()
        deterministic_flag = self.config.is_selectivities_fixed()
        prepp_results = generate_prePP(
            self.plan, "ppmuse", "e", 1, 0, 1, True, self.allPairs, deterministic_flag
        )

        """
        NOTE: From Finn Glück 08.09.2025:
        
        The prepp results need to be updated since they do not consider the costs of sending the events to the cloud.
        The integrated approach considers these costs, so we need to update the prepp results accordingly.
        This is a quick fix to ensure that the results are comparable.
        """
        prepp_results = update_prepp_results(self, prepp_results)
        self.prepp_results_for_debugging = prepp_results

        # Only take the first 4 results to match the schema
        self.results += prepp_results[0:4]


    def _calculate_sum_of_primitive_input_rates_per_query(self) -> Dict:
        """
        Calculate the sum of all primitive input rates for each query.

        For each query, this method identifies all primitive events used in the query
        and sums their global input rates. This value represents the total data volume
        that would be transmitted if all primitive events were pushed to the query
        processing location.

        Returns:
            Dictionary mapping each query object to the sum of its primitive event rates.
            Example: {query1: 1845.0, query2: 1002.0, ...}
        """
        from helper.projString import filter_numbers


        sum_of_input_rates = {}

        for query in self.h_mycombi:
            # Get unique event types (e.g., ['A1', 'A2', 'B'] -> {0, 1} for A and B)
            unique_indices = {
                ord(filter_numbers(event_name)) - ord("A")
                for event_name in query.leafs()
            }

            # Sum rates for unique indices
            sum_of_input_rates[query] = sum(
                self.primitiveEvents[idx]
                for idx in unique_indices
                if idx < len(self.primitiveEvents)
            )

        return sum_of_input_rates

    def _debug_print_all_data(self):
        """Print all available data types and their structures for debugging."""
        import pprint

        pp = pprint.PrettyPrinter(indent=2, width=120, depth=4)

        print("\n" + "=" * 100)
        print("DEBUG: COMPLETE DATA DUMP AT INTEGRATED APPROACH CALL")
        print("=" * 100 + "\n")

        def safe_repr(obj, max_items=5):
            """Safely represent an object with type and content info."""
            obj_type = type(obj).__name__
            if isinstance(obj, (list, tuple)):
                return f"{{{obj_type}: {len(obj)}}} {obj}"
            elif isinstance(obj, dict):
                return f"{{{obj_type}: {len(obj)}}} {obj}"
            elif isinstance(obj, np.ndarray):
                return f"{{{obj_type}: {obj.shape}}} {obj}"
            else:
                return f"{{{obj_type}}} {obj}"

        # Collect all attributes to print
        attrs_to_print = [
            # Configuration
            ("config", self.config),
            ("nwSize", self.nwSize),
            ("node_event_ratio", self.node_event_ratio),
            ("number_eventtypes", self.number_eventtypes),
            ("eventskew", self.eventskew),
            ("max_parents", self.max_parents),
            ("query_size", self.query_size),
            ("query_length", self.query_length),
            ("latency_threshold", self.latency_threshold),
            # Network structure
            ("network", self.network),
            ("root", self.root),
            ("eList", self.eList),
            ("graph", self.graph),
            ("graph_density", self.graph_density),
            ("allPairs", self.allPairs),
            # Event data
            ("eventrates", self.eventrates),
            ("primitiveEvents", self.primitiveEvents),
            # Query and selectivities
            ("query_workload", self.query_workload),
            ("selectivities", self.selectivities),
            ("selectivitiesExperimentData", self.selectivitiesExperimentData),
            ("single_selectivity", self.single_selectivity),
            # Helper variables
            ("h_network_data", self.h_network_data),
            ("h_rates_data", self.h_rates_data),
            ("h_primEvents", self.h_primEvents),
            ("h_instances", self.h_instances),
            ("h_nodes", self.h_nodes),
            ("h_eventNodes", self.h_eventNodes),
            ("h_IndexEventNodes", self.h_IndexEventNodes),
            # Projections
            ("h_projlist", self.h_projlist),
            ("h_projrates", self.h_projrates),
            ("h_projsPerQuery", self.h_projsPerQuery),
            ("h_sharedProjectionsDict", self.h_sharedProjectionsDict),
            ("h_sharedProjectionsList", self.h_sharedProjectionsList),
            ("h_projFilterDict", self.h_projFilterDict),
            # Combigen
            ("h_mycombi", self.h_mycombi),
            ("h_combiDict", self.h_combiDict),
            ("h_criticalMSTypes", self.h_criticalMSTypes),
            ("h_criticalMSProjs", self.h_criticalMSProjs),
            ("h_combiExperimentData", self.h_combiExperimentData),
            (
                "h_primitive_events",
                self.h_primitive_events
                if hasattr(self, "h_primitive_events")
                else None,
            ),
            # Rate lookup and paths
            ("h_local_rate_lookup", self.h_local_rate_lookup),
            ("h_longestPath", self.h_longestPath),
            # Tree and placement
            ("h_treeDict", self.h_treeDict),
            ("h_globalPartitioninInputTypes", self.h_globalPartitioninInputTypes),
            ("h_globalSiSInputTypes", self.h_globalSiSInputTypes),
            ("h_placementTreeDict", self.h_placementTreeDict),
            # Other
            ("networkParams", self.networkParams),
            ("schema", self.schema),
            ("config_single", self.config_single),
            ("CURRENT_SECTION", self.CURRENT_SECTION),
            ("prim", self.prim),
            ("eval_plan", self.eval_plan),
            ("central_eval_plan", self.central_eval_plan),
            ("experiment_result", self.experiment_result),
        ]

        # Print each attribute
        for attr_name, attr_value in attrs_to_print:
            print(f"\n{attr_name} = {safe_repr(attr_value)}")

            # For dicts and lists, print nested structure
            if isinstance(attr_value, dict) and len(attr_value) > 0:
                print(f"  Type: {type(attr_value).__name__}, Length: {len(attr_value)}")
                for i, (key, val) in enumerate(list(attr_value.items())[:5]):
                    if isinstance(val, (dict, list, tuple)):
                        print(f"    {key}: {safe_repr(val)}")
                    else:
                        print(f"    {key}: {val}")
                if len(attr_value) > 5:
                    print(f"    ... ({len(attr_value) - 5} more entries)")

            elif isinstance(attr_value, (list, tuple)) and len(attr_value) > 0:
                print(f"  Type: {type(attr_value).__name__}, Length: {len(attr_value)}")
                for i, item in enumerate(list(attr_value)[:5]):
                    print(f"    [{i}]: {safe_repr(item)}")
                if len(attr_value) > 5:
                    print(f"    ... ({len(attr_value) - 5} more items)")

            elif isinstance(attr_value, np.ndarray):
                print(
                    f"  Type: ndarray, Shape: {attr_value.shape}, Dtype: {attr_value.dtype}"
                )
                print(f"  Data: {attr_value}")

            elif hasattr(attr_value, "__dict__") and attr_name == "config":
                # Special handling for SimulationConfig
                print(f"  Type: {type(attr_value).__name__}")
                for key, val in vars(attr_value).items():
                    print(f"    {key}: {val}")

        # Print additional type information for network nodes
        print("\n--- DETAILED NETWORK NODE INFORMATION ---")
        if self.network:
            for i, node in enumerate(self.network[:3]):
                print(f"\nNode {i}:")
                print(f"  id: {node.id}")
                print(f"  computational_power: {node.computational_power}")
                print(f"  memory: {node.memory}")
                print(f"  eventrates: {node.eventrates}")
                print(
                    f"  Parent: {[p.id if hasattr(p, 'id') else p for p in (node.Parent if node.Parent else [])]}"
                )
                print(
                    f"  Child: {[c.id if hasattr(c, 'id') else c for c in (node.Child if node.Child else [])]}"
                )
            if len(self.network) > 3:
                print(f"\n  ... ({len(self.network) - 3} more nodes)")

        print("\n" + "=" * 100)
        print("END COMPLETE DATA DUMP")
        print("=" * 100 + "\n")

    def _create_optimized_rate_lookup(self) -> Dict[str, Dict[int, float]]:
        """
        Create optimized data structure for O(1) rate lookups.

        Returns:
            Dictionary structure:
            {
                'A': {6: 1000.0, 8: 0.0, 10: 0.0},  # Local rates for event A per node
                'B': {7: 2.0, 10: 0.0},              # Local rates for event B per node
                'C': {7: 45.0, 8: 45.0, 11: 0.0},   # etc.
                ...
            }
        """
        local_rate_lookup = {}

        # For each primitive event type
        for event_type, node_list in self.h_nodes.items():
            local_rate_lookup[event_type] = {}

            # For each node that has this event type
            for node_id in node_list:
                # Calculate local rate for this event at this node
                if node_id < len(self.network):
                    node = self.network[node_id]
                    # Get the index of this event type (A=0, B=1, etc.)
                    event_index = ord(event_type) - ord("A")
                    if event_index < len(node.eventrates):
                        local_rate = float(node.eventrates[event_index])
                        local_rate_lookup[event_type][node_id] = local_rate
                    else:
                        local_rate_lookup[event_type][node_id] = 0.0
                else:
                    local_rate_lookup[event_type][node_id] = 0.0

        return local_rate_lookup

    def _initialize_network_topology(self):
        """Create network topology based on configuration mode."""
        if self.config.is_topology_fixed():
            self.root, self.network, self.eList = create_hardcoded_tree()
        else:
            self.root, self.network, self.eList = create_random_tree(
                self.nwSize, self.eventrates, self.node_event_ratio, self.max_parents
            )

    def _initialize_network_graph(self):
        """Initialize network graph and distance calculations."""
        self.graph = create_fog_graph(self.network)
        self.allPairs = populate_allPairs(self.graph)
        self.h_longestPath = getLongest(self.allPairs)

    def _initialize_query_workload(self):
        """Generate query workload based on configuration mode."""
        if self.config.is_workload_fixed():
            self.query_workload = generate_hardcoded_workload()
        else:
            self.query_workload = generate_workload(
                self.query_size, self.query_length, self.primitiveEvents
            )

    def _initialize_selectivities(self):
        """Initialize selectivities based on configuration mode."""
        if self.config.is_selectivities_fixed():
            self.selectivities, self.selectivitiesExperimentData = (
                generate_hardcoded_selectivities()
            )
        else:
            self.selectivities, self.selectivitiesExperimentData = (
                initialize_selectivities(self.primitiveEvents)
            )
