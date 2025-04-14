import time
import matplotlib.pyplot as plt
from Node import Node
from network import generate_eventrates, create_random_tree,generate_events
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
from projections import generate_all_projections


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
    h_combiDict = None
    h_criticalMSTypes_criticalMSProjs = None
    h_combiExperimentData = None
    h_criticalMSTypes = None
    h_criticalMSProjs = None
    h_combiDict = {}
    h_globalPartitioninInputTypes = {}
    h_globalSiSInputTypes = {}

    

    def __init__(self, nwSize: int, node_event_ratio: float, num_eventtypes: int, eventskew: float, max_partens: int, query_size: int, query_length:int):
        self.schema = ["nwSize, node_event_ratio, num_eventtypes, eventskew, max_parents, query_size, query_length"] 
        self.nwSize = nwSize
        self.node_event_ratio = node_event_ratio
        self.number_eventtypes = num_eventtypes
        self.eventskew = eventskew
        self.max_parents = max_partens
        self.query_size = query_size
        self.query_length = query_length

        # Add a dictionary to store the times of each function call
        self.function_times = {}

        # Use time to benchmark each step
        start_time = time.time()
        self.eventrates = generate_eventrates(eventskew,num_eventtypes)
        self.function_times["Functions"] = time.time() - start_time

        start_time = time.time()
        self.networkParams = [self.eventskew, self.number_eventtypes, self.node_event_ratio, self.nwSize, min(self.eventrates)/max(self.eventrates)]
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.primitiveEvents = generate_events(self.eventrates, node_event_ratio)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        root, self.network = create_random_tree(nwSize, self.eventrates, node_event_ratio, max_partens)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.graph = create_fog_graph(self.network)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.allPairs = populate_allPairs(self.graph)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.h_longestPath = getLongest(self.allPairs)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.query_workload = generate_workload(query_size, query_length, self.primitiveEvents)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.selectivities, self.selectivitiesExperimentData = initialize_selectivities(self.primitiveEvents)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.config_single = generate_config_buffer(self.network, self.query_workload, self.selectivities)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.single_selectivity = initializeSingleSelectivity(self.CURRENT_SECTION, self.config_single, self.query_workload)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.h_network_data, self.h_rates_data, self.h_primEvents, self.h_instances, self.h_nodes = initialize_globals(self.network)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.h_eventNodes, self.h_IndexEventNodes = initEventNodes(self.h_nodes, self.h_network_data)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.h_projlist, self.h_projrates, self.h_projsPerQuery, self.h_sharedProjectionsDict, self.h_sharedProjectionsList = generate_all_projections(self)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.h_projFilterDict = populate_projFilterDict(self)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.h_projFilterDict = removeFilters(self)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.h_mycombi, self.h_combiDict, self.h_criticalMSTypes_criticalMSProjs, self.h_combiExperimentData = generate_combigen(self)
        self.function_times["INEv"] = time.time() - start_time

        start_time = time.time()
        self.h_criticalMSTypes, self.h_criticalMSProjs = self.h_criticalMSTypes_criticalMSProjs
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.eval_plan, self.central_eval_plan, self.experiment_result, self.results = calculate_operatorPlacement(self, 'test', self.max_parents)
        self.function_times["INEv"] += time.time() - start_time

        start_time = time.time()
        self.plan = generate_eval_plan(self.network, self.selectivities, self.eval_plan, self.central_eval_plan, self.query_workload)
        self.function_times["Functions"] += time.time() - start_time

        start_time = time.time()
        self.results += generate_prePP(self.plan, 'ppmuse', 'e', 0, 0, 1, False, self.allPairs)
        self.function_times["PrePP"] = time.time() - start_time

    def plot_benchmark_times(self):
    # Zusammenfassen der Zeiten pro Kategorie
        category_times = {
            "Functions": self.function_times.get("Functions", 0),
            "INEv": self.function_times.get("INEv", 0),
            "PrePP": self.function_times.get("PrePP", 0)
        }

        categories = list(category_times.keys())
        times = list(category_times.values())
        colors = ["#4CAF50", "#2196F3", "#FF9800"]  # Grün, Blau, Orange

        plt.figure(figsize=(10, 6))
        bars = plt.barh(categories, times, color=colors)

        # Beschriftung der Balken mit den Werten
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{width:.2f}s", va='center')

        plt.xlabel("Execution Time (seconds)")
        plt.ylabel("Component")
        plt.title("Benchmark: Execution Time per Pipeline Component")

        # Zusatzinfo unten
        info_text = f"Nodes: {self.nwSize} | Max Parents: {self.max_parents}"
        plt.figtext(0.5, 0.01, info_text, wrap=True, horizontalalignment='center', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 1])  # Platz unten lassen für Info
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        plt.show()

my_ines = INES(50, 0.5, 6, 0.3, 20, 5, 10)
my_ines.plot_benchmark_times()
