from Node import Node
from network import generate_eventrates, create_random_tree, generate_events
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
from generateEvalPlan import generate_eval_plan
from prepp import generate_prePP


class INES():
    # Core system components
    allPairs: list
    network: list[Node]
    eventrates: list[list[int]]
    query_workload = None
    selectivities = None
    selectivitiesExperimentData = None
    primitiveEvents: list[int]
    config_single: None
    single_selectivity = None

    # Network parameters
    nwSize: int
    node_event_ratio: float
    number_eventtypes: int
    eventskew: float
    max_parents: int
    query_size: int
    query_length: int
    networkParams: dict

    # Optimization results
    eval_plan = None
    central_eval_plan = None
    experiment_result = None
    prim = None
    CURRENT_SECTION = ''

    # Helper Variables from different Files - namespace management
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
    h_criticalMSTypes_criticalMSProjs = None
    h_combiExperimentData = None
    h_criticalMSTypes = None
    h_criticalMSProjs = None
    h_combiDict = {}
    h_globalPartitioninInputTypes = {}
    h_globalSiSInputTypes = {}
    h_placementTreeDict = {}

    def __init__(self, network_size: int, node_event_ratio: float, num_eventtypes: int,
                 eventskew: float, max_parents: int, num_of_queries: int, query_length: int, use_deterministic_scenario: bool = False):

        # Define the results schema for data collection
        self.schema = ["ID", "TransmissionRatio", "Transmission", "INEvTransmission", "FilterUsed", "Nodes",
                       "EventSkew", "EventNodeRatio", "WorkloadSize", "NumberProjections", "MinimalSelectivity",
                       "MedianSelectivity", "CombigenComputationTime", "Efficiency", "PlacementComputationTime",
                       "centralHopLatency", "Depth", "CentralTransmission", "LowerBound", "EventTypes",
                       "MaximumParents", "exact_costs", "PushPullTime", "MaxPushPullLatency"]

        # Store initialization parameters
        self.nwSize = network_size
        self.node_event_ratio = node_event_ratio
        self.number_eventtypes = num_eventtypes
        self.eventskew = eventskew
        self.max_parents = max_parents
        self.query_size = num_of_queries
        self.query_length = query_length

        # Import here to avoid circular dependencies
        from projections import generate_all_projections

        print("[INES] Initializing Intelligent Network Event System...")
        
        if use_deterministic_scenario:
            print(f"[INES] Configuration mode: Deterministic scenario (reproduces exact results)")
            from joint_optimization.hardcoded_topology.hardcoded_scenario import (create_hardcoded_network, create_hardcoded_queries,
                                                                                      get_hardcoded_eventrates, get_hardcoded_primitive_events)
            
            # Use exact values from the successful simulation run
            print("[INES] Step 1/18: Loading deterministic event rates...")
            self.eventrates = get_hardcoded_eventrates()
            print(f"[INES] Loaded event rates for {len(self.eventrates)} event types: {self.eventrates}")

            print("[INES] Step 2/18: Loading deterministic primitive events...")
            self.primitiveEvents = get_hardcoded_primitive_events()
            print(f"[INES] Loaded {len(self.primitiveEvents)} primitive events: {self.primitiveEvents}")

            print("[INES] Step 3/18: Creating deterministic network topology...")
            root, self.network = create_hardcoded_network()
            print("[INES] Loaded exact 12-node topology from successful simulation")
            
        else:
            print(f"[INES] Configuration mode: Random topology")

            # Step 1: Generate realistic event rates using Zipf distribution
            # This stays normal - we want realistic event characteristics
            print("[INES] Step 1/18: Generating event rates using Zipf distribution...")
            self.eventrates = generate_eventrates(eventskew, num_eventtypes)
            print(f"[INES] Generated event rates for {num_eventtypes} event types: {self.eventrates}")

            # Step 2: Generate primitive events based on node_event_ratio
            # This also stays normal - we want realistic event distribution
            print("[INES] Step 2/18: Generating primitive events distribution...")
            self.primitiveEvents = generate_events(self.eventrates, node_event_ratio)
            print(f"[INES] Generated {len(self.primitiveEvents)} primitive events: {self.primitiveEvents}")

            # Step 3: Create network topology - HERE is where we make our choice
            print("[INES] Step 3/18: Creating network topology...")

            print("[INES] Generating random tree topology")
            root, self.network = create_random_tree(network_size, self.eventrates,
                                                        node_event_ratio, max_parents)

        # Print the actual network that was created
        self._print_network_summary()

        # Step 4: Calculate network parameters for analysis
        print("[INES] Step 4/18: Calculating network parameters for analysis...")
        self.networkParams = {
            "eventskew": self.eventskew,
            "number_eventtypes": self.number_eventtypes,
            "node_event_ratio": self.node_event_ratio,
            "network_size": self.nwSize,
            "min_eventrate/max_eventrate": min(self.eventrates) / max(self.eventrates)
        }

        # Step 5: Create the network graph representation
        print("[INES] Step 5/18: Creating network graph representation...")
        self.graph = create_fog_graph(self.network)

        # Step 6: Calculate all-pairs shortest paths for routing optimization
        print("[INES] Step 6/18: Computing all-pairs shortest paths for routing optimization...")
        self.allPairs = populate_allPairs(self.graph)
        self.h_longestPath = getLongest(self.allPairs)
        print(f"[INES] Network analysis complete - longest path: {self.h_longestPath} hops")

        # Step 7: Generate query workload
        print("[INES] Step 7/18: Generating query workload...")
        if use_deterministic_scenario:
            from joint_optimization.hardcoded_topology.hardcoded_scenario import create_hardcoded_queries
            self.query_workload = create_hardcoded_queries()
            print(f"[INES] Loaded deterministic workload with {len(self.query_workload)} queries:")
        else:
            self.query_workload = generate_workload(num_of_queries, query_length, self.primitiveEvents)
            print(f"[INES] Generated workload with {len(self.query_workload)} queries:")
        
        for i, query in enumerate(self.query_workload):
            print(f"[INES]   Query {i + 1}: {str(query)} (events: {query.leafs()})")

        # Step 8: Initialize selectivities - normal INES behavior
        print("[INES] Step 8/18: Initializing event selectivities...")
        self.selectivities, self.selectivitiesExperimentData = initialize_selectivities(self.primitiveEvents)

        # Step 9: Generate configuration buffer for single selectivities
        print("[INES] Step 9/18: Generating configuration buffer for single selectivities...")
        self.config_single = generate_config_buffer(self.network, self.query_workload, self.selectivities)

        # Step 10: Initialize single selectivities - this is where the original error occurred
        print("[INES] Step 10/18: Initializing single selectivities...")
        self.single_selectivity = initializeSingleSelectivity(self.CURRENT_SECTION, self.config_single,
                                                              self.query_workload)

        # Step 11: Initialize global helper variables
        print("[INES] Step 11/18: Initializing global helper variables...")
        self.h_network_data, self.h_rates_data, self.h_primEvents, self.h_instances, self.h_nodes = initialize_globals(
            self.network)

        # Step 12: Initialize event tracking structures
        print("[INES] Step 12/18: Initializing event tracking structures...")
        self.h_eventNodes, self.h_IndexEventNodes = initEventNodes(self.h_nodes, self.h_network_data)

        # Step 13: Generate all possible projections - INES intelligence at work
        print("[INES] Step 13/18: Generating all possible beneficial projections...")
        self.h_projlist, self.h_projrates, self.h_projsPerQuery, self.h_sharedProjectionsDict, self.h_sharedProjectionsList = generate_all_projections(
            self)
        print(f"[INES] Intelligence analysis complete - generated {len(self.h_projlist)} beneficial projections")

        # Step 14: Set up filter dictionaries
        print("[INES] Step 14/18: Setting up projection filter dictionaries...")
        self.h_projFilterDict = populate_projFilterDict(self)
        self.h_projFilterDict = removeFilters(self)

        # Step 15: Generate optimal combinations - this is where INES gets really clever
        print("[INES] Step 15/18: Generating optimal projection combinations...")
        self.h_mycombi, self.h_combiDict, self.h_criticalMSTypes_criticalMSProjs, self.h_combiExperimentData = generate_combigen(
            self)
        self.h_criticalMSTypes, self.h_criticalMSProjs = self.h_criticalMSTypes_criticalMSProjs
        print(f"[INES] Optimization intelligence - found optimal combinations for {len(self.h_mycombi)} projections")

        # Step 16: Calculate operator placement - the heart of the optimization
        print("[INES] Step 16/18: Computing optimal operator placement...")
        self.eval_plan, self.central_eval_plan, self.experiment_result, self.results = calculate_operatorPlacement(
            self,
            'test',
            self.max_parents)

        # Step 17: Generate evaluation plan
        print("[INES] Step 17/18: Generating evaluation plan for network deployment...")
        self.plan = generate_eval_plan(self.network, self.selectivities, self.eval_plan, self.central_eval_plan,
                                       self.query_workload)

        # Step 18: Optimize communication with push-pull strategies
        print("[INES] Step 18/18: Optimizing communication with push-pull strategies...")
        self.results += generate_prePP(self.plan, 'ppmuse', 'e', 0, 0, 1, False, self.allPairs)

        # Final results summary
        print("[INES] System initialization completed successfully!")
        self._print_optimization_results()


    def _print_network_summary(self):
        """
        Print a comprehensive summary of the created network.

        This helps us understand what network INES is working with and
        makes it easier to predict and understand the optimization decisions.
        """
        print("\n" + "=" * 60)
        print("[INES] NETWORK TOPOLOGY SUMMARY")
        print("=" * 60)

        print("\n[INES] Event Distribution Across Nodes:")
        for node in self.network:
            events_produced = [(j, node.eventrates[j]) for j in range(len(node.eventrates)) if node.eventrates[j] > 0]
            if events_produced:
                event_info = [f"{chr(ord('A') + j)}:{rate}" for j, rate in events_produced]
                print(f"[INES]   Node {node.id}: {', '.join(event_info)}")
            else:
                print(f"[INES]   Node {node.id}: No events produced")

        print(f"\n[INES] Network Topology Connections:")
        for node in self.network:
            if node.Parent:
                parent_ids = [p.id for p in node.Parent]
                print(f"[INES]   Node {node.id} -> Parents: {parent_ids}")
            if node.Child:
                child_ids = [c.id for c in node.Child]
                print(f"[INES]   Node {node.id} -> Children: {child_ids}")

        print("=" * 60 + "\n")

    def _print_optimization_results(self):
        """
        Print the final optimization results in an easy-to-understand format.

        This shows us how well INES performed on our controlled topology
        and helps us understand the effectiveness of its decisions.
        """
        if hasattr(self, 'results') and len(self.results) > 1:
            transmission_ratio = self.results[1]
            total_costs = self.results[2]
            optimized_costs = self.results[3]

            print("\n" + "=" * 60)
            print("[INES] OPTIMIZATION RESULTS")
            print("=" * 60)
            print(f"[INES] Transmission Ratio: {transmission_ratio:.3f}")
            if transmission_ratio < 1.0:
                savings = (1.0 - transmission_ratio) * 100
                print(f"[INES] Network Traffic Reduction: {savings:.1f}%")
                print(f"[INES] Central Processing Cost: {total_costs:.0f} units")
                print(f"[INES] Optimized INES Cost: {optimized_costs:.0f} units")
                print(f"[INES] Absolute Cost Savings: {total_costs - optimized_costs:.0f} units")
            print("=" * 60 + "\n")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()