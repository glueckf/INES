from Node import Node
from network import generate_eventrates, create_random_tree, generate_events
from graph import create_fog_graph
from graph import draw_graph
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
import csv
import math


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
    networkParams: list

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
                 eventskew: float, max_parents: int, num_of_queries: int, query_length: int,
                 use_hardcoded_topology: bool = False):

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
        print(f"[INES] Configuration mode: {'Hardcoded topology' if use_hardcoded_topology else 'Random topology'}")

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
        if use_hardcoded_topology:
            print("[INES] Building hardcoded 5-node fog-cloud tree topology")
            root, self.network = self._create_hardcoded_network()
            # Assign events to our hardcoded network using normal INES logic
            self._assign_events_to_hardcoded_network()
        else:
            print("[INES] Generating random tree topology")
            root, self.network = create_random_tree(network_size, self.eventrates,
                                                    node_event_ratio, max_parents)

        # Print the actual network that was created
        self._print_network_summary()

        # Step 4: Calculate network parameters for analysis
        print("[INES] Step 4/18: Calculating network parameters for analysis...")
        self.networkParams = [self.eventskew, self.number_eventtypes, self.node_event_ratio,
                              self.nwSize, min(self.eventrates) / max(self.eventrates)]

        # Step 5: Create the network graph representation
        print("[INES] Step 5/18: Creating network graph representation...")
        self.graph = create_fog_graph(self.network)

        # Step 6: Calculate all-pairs shortest paths for routing optimization
        print("[INES] Step 6/18: Computing all-pairs shortest paths for routing optimization...")
        self.allPairs = populate_allPairs(self.graph)
        self.h_longestPath = getLongest(self.allPairs)
        print(f"[INES] Network analysis complete - longest path: {self.h_longestPath} hops")

        # Step 7: Generate query workload - this stays normal for realistic queries
        print("[INES] Step 7/18: Generating query workload...")
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
        self.eval_plan, self.central_eval_plan, self.experiment_result, self.results = calculate_operatorPlacement(self,
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

    def _create_hardcoded_network(self):
        """
        Create our controlled 5-node fog-cloud tree for experiments.

        This creates just the topology structure - events will be assigned normally
        by INES using the standard algorithms. This gives us predictable structure
        but realistic event distribution.

        Network Structure:
                Node 0 (Cloud)
               /              \
           Node 1 (Fog)    Node 2 (Fog)
           /                    \
        Node 3 (Edge)        Node 4 (Edge)

        Returns:
            Tuple of (root_node, network_list)
        """
        network = []

        # Node 0: Cloud/Root node with infinite resources
        root = Node(0, math.inf, math.inf)
        network.append(root)

        # Node 1: Fog node with medium resources
        node1 = Node(1, 5, 5)
        node1.Parent.append(root)
        root.Child.append(node1)
        network.append(node1)

        # Node 2: Fog node with medium resources
        node2 = Node(2, 5, 5)
        node2.Parent.append(root)
        root.Child.append(node2)
        network.append(node2)

        # Node 3: Edge node with limited resources
        node3 = Node(3, 1, 1)
        node3.Parent.append(node1)
        node1.Child.append(node3)
        network.append(node3)

        # Node 4: Edge node with limited resources
        node4 = Node(4, 1, 1)
        node4.Parent.append(node2)
        node2.Child.append(node4)
        network.append(node4)

        print(f"[INES] Created controlled topology: Cloud(0) -> Fog(1,2) -> Edge(3,4)")
        return root, network

    def _assign_events_to_hardcoded_network(self):
        """
        Assign events to our hardcoded network using normal INES logic.

        This is the key insight - we use the standard INES event assignment
        algorithm on our controlled topology. This means:
        - Leaf nodes (3, 4) will get events based on node_event_ratio
        - Event types will be distributed according to the eventrates
        - Everything will be realistic but on our predictable structure
        """
        from network import _assign_event_rates
        print("[INES] Assigning events to hardcoded topology using standard INES algorithm...")
        _assign_event_rates(self.network, self.eventrates, self.node_event_ratio)

        # Ensure every event type has at least one producer (INES requirement)
        self._ensure_all_events_have_producers()

    def _ensure_all_events_have_producers(self):
        """
        Make sure every event type has at least one producer.

        This is important because INES expects all event types to be available
        somewhere in the network. If random assignment misses an event type,
        we assign it to a random leaf node.
        """
        import pandas as pd
        import random

        # Find leaf nodes
        leaf_nodes = [node for node in self.network if len(node.Child) == 0]

        # Check which events are missing
        event_sums = pd.DataFrame([node.eventrates for node in leaf_nodes])

        for event_idx in range(len(self.eventrates)):
            if event_sums[event_idx].sum() == 0:
                # This event type has no producer - assign it to a random leaf
                random_leaf = random.choice(leaf_nodes)
                random_leaf.eventrates[event_idx] = self.eventrates[event_idx]
                print(f"[INES] Assigned missing event type {event_idx} to leaf node {random_leaf.id}")

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

    def return_selectivity(self, proj):
        """
        Return selectivity for arbitrary projection.

        This is a helper function used throughout INES to calculate
        how much data reduction a projection provides.
        """
        from helper.projString import filter_numbers
        import helper.subsets as sbs

        selectivities = self.selectivities
        proj = list(map(lambda x: filter_numbers(x), proj))
        two_temp = sbs.printcombination(proj, 2)
        selectivity = 1
        for two_s in two_temp:
            if two_s in selectivities.keys():
                if selectivities[two_s] != 1:
                    selectivity *= selectivities[two_s]
        return selectivity


def main():
    """
    Main function demonstrating both random and hardcoded topology modes.

    This is where you can easily switch between controlled experiments
    and normal INES operation. The hardcoded topology is perfect for
    learning and understanding, while random topology shows real-world performance.
    """
    try:
        print("=" * 80)
        print("[MAIN] INES - Intelligent Network Event System")
        print("=" * 80)

        # Configuration - easily switch between modes here
        USE_HARDCODED_TOPOLOGY = False  # Set to False for random networks

        if USE_HARDCODED_TOPOLOGY:
            print("\n[MAIN] Running in CONTROLLED EXPERIMENT mode")
            print("[MAIN] Using hardcoded 5-node topology for predictable learning")

            my_ines = INES(
                network_size=5,  # Our controlled 5-node tree
                node_event_ratio=0.5,  # 50% chance leaf nodes generate events
                num_eventtypes=6,  # Enough event types for interesting queries
                eventskew=0.3,  # Moderate skew in event rates
                max_parents=1,  # Simple tree structure (ignored for hardcoded)
                num_of_queries=3,  # Manageable number of queries to understand
                query_length=4,  # Complex enough queries to see optimization
                use_hardcoded_topology=True
            )
        else:
            print("\n[MAIN] Running in NORMAL mode")
            print("[MAIN] Using random network generation for realistic scenarios")

            my_ines = INES(
                network_size=12,  # Larger network for real-world simulation
                node_event_ratio=0.5,
                num_eventtypes=6,
                eventskew=0.3,
                max_parents=10,  # More complex topology possible
                num_of_queries=3,
                query_length=5,
                use_hardcoded_topology=False
            )

        print("\n[MAIN] Success! INES has optimized your event processing system.")
        print(f"[MAIN] Network topology: {len(my_ines.network)} nodes")
        print(f"[MAIN] Query workload: {len(my_ines.query_workload)} queries optimized")
        print(f"[MAIN] Intelligence layer: {len(my_ines.h_projlist)} beneficial projections generated")

        # Encourage experimentation
        if USE_HARDCODED_TOPOLOGY:
            print("\n[MAIN] Experiment Ideas for Learning:")
            print("[MAIN]   - Modify event rates to observe placement adaptation")
            print("[MAIN]   - Alter topology structure to study routing optimization")
            print("[MAIN]   - Increase event types to explore complexity scaling")
            print("[MAIN]   - Test different query patterns for optimization insights")

    except Exception as e:
        print(f"\n[ERROR] INES initialization failed: {str(e)}")
        import traceback
        print("\n[ERROR] Full error trace:")
        traceback.print_exc()

        print("\n[ERROR] Debugging suggestions:")
        print("[ERROR]   1. Verify all required modules are imported correctly")
        print("[ERROR]   2. Check network topology connections are valid")
        print("[ERROR]   3. Ensure event rates are properly assigned to nodes")
        print("[ERROR]   4. Confirm query workload uses available event types")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()