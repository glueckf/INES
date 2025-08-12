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

    

    def __init__(self, nwSize: int, node_event_ratio: float, num_eventtypes: int, eventskew: float, max_partens: int, query_size: int, query_length:int):
        self.schema = ["ID", "TransmissionRatio", "Transmission","INEvTransmission","FilterUsed", "Nodes", "EventSkew", "EventNodeRatio", "WorkloadSize", "NumberProjections", "MinimalSelectivity", "MedianSelectivity","CombigenComputationTime", "Efficiency", "PlacementComputationTime", "centralHopLatency", "Depth",  "CentralTransmission", "LowerBound", "EventTypes", "MaximumParents", "exact_costs","PushPullTime","MaxPushPullLatency"] 
        self.nwSize = nwSize
        self.node_event_ratio = node_event_ratio
        self.number_eventtypes = num_eventtypes
        self.eventskew = eventskew
        self.max_parents = max_partens
        self.query_size = query_size
        self.query_length = query_length

        from projections import generate_all_projections
        self.eventrates = generate_eventrates(eventskew,num_eventtypes)
        self.networkParams = [self.eventskew,self.number_eventtypes,self.node_event_ratio,self.nwSize,min(self.eventrates)/max(self.eventrates)]
        self.primitiveEvents= generate_events(self.eventrates,node_event_ratio)
        self.root, self.network, self.eList = create_random_tree(nwSize,self.eventrates,node_event_ratio,max_partens) 
        self.graph = create_fog_graph(self.network)
        self.allPairs = populate_allPairs(self.graph)
        self.h_longestPath = getLongest(self.allPairs)
        self.query_workload = generate_workload(query_size,query_length,self.primitiveEvents)
        self.selectivities,self.selectivitiesExperimentData = initialize_selectivities(self.primitiveEvents)
        self.config_single = generate_config_buffer(self.network,self.query_workload,self.selectivities)
        self.single_selectivity = initializeSingleSelectivity(self.CURRENT_SECTION, self.config_single, self.query_workload)

        #This is important to variious files afterwards
        self.h_network_data,self.h_rates_data,self.h_primEvents,self.h_instances,self.h_nodes = initialize_globals(self.network)
        #print(f"DATA {self.h_network_data} and NETWORK {self.h_nodes}")
        self.h_eventNodes,self.h_IndexEventNodes = initEventNodes(self.h_nodes,self.h_network_data)
        # treeDict for graph compression
        self.h_treeDict = treeDict(self.h_network_data, self.eList)
        #print(f"treeDict{self.h_treeDict}")
        # call graph compression function
        self.graph = compressed_graph(self.graph, self.h_treeDict)
        self.h_projlist,self.h_projrates,self.h_projsPerQuery,self.h_sharedProjectionsDict,self.h_sharedProjectionsList = generate_all_projections(self)
        self.h_projFilterDict = populate_projFilterDict(self)
        self.h_projFilterDict= removeFilters(self)
        self.h_mycombi, self.h_combiDict,self.h_criticalMSTypes_criticalMSProjs, self.h_combiExperimentData = generate_combigen(self)
        self.h_criticalMSTypes, self.h_criticalMSProjs = self.h_criticalMSTypes_criticalMSProjs
        self.eval_plan,self.central_eval_plan,self.experiment_result,self.results = calculate_operatorPlacement(self,'test',self.max_parents)
        self.plan=generate_eval_plan(self.network,self.selectivities,self.eval_plan,self.central_eval_plan,self.query_workload)
        self.results += generate_prePP(self.plan,'ppmuse','e',0,0,1,False,self.allPairs)
        # new =False
        # try:
        #      f = open("./res/"+str(filename)+".csv")   
        # except FileNotFoundError:
        #      new = True           
        #      with open("./res/"+str(filename)+".csv", "w")as f:
        #          pass
             
        # with open("./res/"+str(filename)+".csv", "a") as result:
        #    writer = csv.writer(result)  
        #    if new:
        #        writer.writerow(self.schema)              
        #    writer.writerow(self.results)
        
        
        

# import traceback
# import logging

# # Set up logging to capture all errors in a file
# logging.basicConfig(
#     filename="error_log.txt",
#     level=logging.ERROR,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

try:
    my_ines = INES(50, 0.5, 8, 1.3, 10, 5, 5)
except Exception as e:
    error_message = f"❌ Exception: {str(e)}\n"
    print(error_message)  # Optional: also print to console
