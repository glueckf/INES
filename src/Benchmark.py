import time
import matplotlib.pyplot as plt
from Node import Node
from network import generate_eventrates, create_random_tree,generate_events, compressed_graph, treeDict
from graph import create_fog_graph
from graph import draw_graph
from allPairs import populate_allPairs
from queryworkload import generate_workload
from selectivity import initialize_selectivities
from write_config_single import generate_config_buffer
from singleSelectivities import initializeSingleSelectivity
from helper.parse_network import initialize_globals
from helper.structures import initEventNodes
from combigen import populate_projFilterDict,removeFilters,generate_combigen
from helper.structures import getLongest
from operatorplacement import calculate_operatorPlacement
from generateEvalPlan import generate_eval_plan
from prepp import generate_prePP
import csv

class INES():
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
    query_size:int 
    query_length: int
    networkParams: list
    eval_plan = None
    central_eval_plan = None
    experiment_result = None
    prim = None
    CURRENT_SECTION = ''
    eList = {}
    h_treeDict = {}
    
    "Helper Variables from different Files - namespace issues"
    h_network_data = None
    h_rates_data = None
    h_primEvents = None
    h_instances=None
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
    #h_combiDict = None
    h_criticalMSTypes_criticalMSProjs = None
    h_combiExperimentData = None
    h_criticalMSTypes = None
    h_criticalMSProjs = None
    h_combiDict = {}
    h_globalPartitioninInputTypes = {}
    h_globalSiSInputTypes = {}
    h_placementTreeDict = {}

    

    def __init__(self, nwSize: int, node_event_ratio: float, num_eventtypes: int, eventskew: float, max_parents: int, query_size: int, query_length:int):
        self.schema = ["generate_eventrates", "networkParams", "generate_events", "create_random_tree", "create_fog_graph", "populate_allPairs", "getLongest", "generate_workload", "initialize_selectivities", "generate_config_buffer", "initializeSingleSelectivity", "initialize_globals", "initEventNodes", "treeDict", "compressed_graph", "generate_all_projections", "populate_projFilterDict", "removeFilters", "generate_combigen", "set_criticalMSTypes", "calculate_operatorPlacement", "generate_eval_plan", "generate_prePP"] #"treeDict", "compressed_graph",
        #self.schema = ["ID", "TransmissionRatio", "Transmission","INEvTransmission","FilterUsed", "Nodes", "EventSkew", "EventNodeRatio", "WorkloadSize", "NumberProjections", "MinimalSelectivity", "MedianSelectivity","CombigenComputationTime", "Efficiency", "PlacementComputationTime", "centralHopLatency", "Depth",  "CentralTransmission", "LowerBound", "EventTypes", "MaximumParents", "exact_costs","PushPullTime","MaxPushPullLatency"] 
        self.nwSize = nwSize
        self.node_event_ratio = node_event_ratio
        self.number_eventtypes = num_eventtypes
        self.eventskew = eventskew
        self.max_parents = max_parents
        self.query_size = query_size
        self.query_length = query_length

        from projections import generate_all_projections

        # Add a dictionary to store the times of each function call
        self.function_times = {}

        # Use time to benchmark each step
        start_time = time.time()
        self.eventrates = generate_eventrates(eventskew,num_eventtypes)
        self.function_times["generate_eventrates"] = time.time() - start_time

        start_time = time.time()
        self.networkParams = [self.eventskew, self.number_eventtypes, self.node_event_ratio, self.nwSize, min(self.eventrates)/max(self.eventrates)]
        self.function_times["networkParams"] = time.time() - start_time

        start_time = time.time()
        self.primitiveEvents = generate_events(self.eventrates, node_event_ratio)
        self.function_times["generate_events"] = time.time() - start_time

        start_time = time.time()
        self.root, self.network, self.eList = create_random_tree(nwSize, self.eventrates, node_event_ratio, max_parents)
        self.function_times["create_random_tree"] = time.time() - start_time

        start_time = time.time()
        self.graph = create_fog_graph(self.network)
        self.function_times["create_fog_graph"] = time.time() - start_time

        start_time = time.time()
        self.allPairs = populate_allPairs(self.graph)
        self.function_times["populate_allPairs"] = time.time() - start_time

        start_time = time.time()
        self.h_longestPath = getLongest(self.allPairs)
        self.function_times["getLongest"] = time.time() - start_time

        start_time = time.time()
        self.query_workload = generate_workload(query_size, query_length, self.primitiveEvents)
        self.function_times["generate_workload"] = time.time() - start_time

        start_time = time.time()
        self.selectivities, self.selectivitiesExperimentData = initialize_selectivities(self.primitiveEvents)
        self.function_times["initialize_selectivities"] = time.time() - start_time

        start_time = time.time()
        self.config_single = generate_config_buffer(self.network, self.query_workload, self.selectivities)
        self.function_times["generate_config_buffer"] = time.time() - start_time

        start_time = time.time()
        self.single_selectivity = initializeSingleSelectivity(self.CURRENT_SECTION, self.config_single, self.query_workload)
        self.function_times["initializeSingleSelectivity"] = time.time() - start_time

        start_time = time.time()
        self.h_network_data, self.h_rates_data, self.h_primEvents, self.h_instances, self.h_nodes = initialize_globals(self.network)
        self.function_times["initialize_globals"] = time.time() - start_time

        start_time = time.time()
        self.h_eventNodes, self.h_IndexEventNodes = initEventNodes(self.h_nodes, self.h_network_data)
        self.function_times["initEventNodes"] = time.time() - start_time

        start_time = time.time()
        self.h_treeDict = treeDict(self.h_network_data, self.eList)
        self.function_times["treeDict"] = time.time() - start_time

        start_time = time.time()
        self.graph = compressed_graph(self.graph, self.h_treeDict)
        self.function_times["compressed_graph"] = time.time() - start_time

        start_time = time.time()
        self.h_projlist, self.h_projrates, self.h_projsPerQuery, self.h_sharedProjectionsDict, self.h_sharedProjectionsList = generate_all_projections(self)
        self.function_times["generate_all_projections"] = time.time() - start_time

        start_time = time.time()
        self.h_projFilterDict = populate_projFilterDict(self)
        self.function_times["populate_projFilterDict"] = time.time() - start_time

        start_time = time.time()
        self.h_projFilterDict = removeFilters(self)
        self.function_times["removeFilters"] = time.time() - start_time

        start_time = time.time()
        self.h_mycombi, self.h_combiDict, self.h_criticalMSTypes_criticalMSProjs, self.h_combiExperimentData = generate_combigen(self)
        self.function_times["generate_combigen"] = time.time() - start_time

        start_time = time.time()
        self.h_criticalMSTypes, self.h_criticalMSProjs = self.h_criticalMSTypes_criticalMSProjs
        self.function_times["set_criticalMSTypes"] = time.time() - start_time

        start_time = time.time()
        self.eval_plan, self.central_eval_plan, self.experiment_result, self.results = calculate_operatorPlacement(self, 'test', self.max_parents)
        self.function_times["calculate_operatorPlacement"] = time.time() - start_time

        start_time = time.time()
        self.plan = generate_eval_plan(self.network, self.selectivities, self.eval_plan, self.central_eval_plan, self.query_workload)
        self.function_times["generate_eval_plan"] = time.time() - start_time

        start_time = time.time()
        self.results += generate_prePP(self.plan, 'ppmuse', 'e', 0, 0, 1, False, self.allPairs)
        self.function_times["generate_prePP"] = time.time() - start_time

# my_ines = INES(12, 0.5, 6, 0.3, 4, 5, 10)
# my_ines.plot_benchmark_times()

# if __name__ == "__main__":
#     all_function_times = {"Functions": [], "INEv": [], "PrePP": []}

#     for i in range(2):
#         print(f"🔁 Run {i+1}/{10}")
#         try:
#             my_ines = INES(70, 0.5, 6, 0.3, 10, 6, 8)
#             all_function_times["Functions"].append(my_ines.function_times.get("Functions", 0))
#             all_function_times["INEv"].append(my_ines.function_times.get("INEv", 0))
#             all_function_times["PrePP"].append(my_ines.function_times.get("PrePP", 0))
#         except Exception as e:
#             print(f"❌ Run {i+1} failed: {e}")

#     # Call avg plotting
#     if 'my_ines':  # only if at least one run succeeded
#         my_ines.avg_function_times(all_function_times)
