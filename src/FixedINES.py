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
from queryworkload import AND, SEQ, PrimEvent
import numpy as np

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
    h_projFilterDict = {}
    h_longestPath = None
    h_mycombi = None
    #h_combiDict = None
    h_criticalMSTypes_criticalMSProjs = None
    h_combiExperimentData = None
    h_criticalMSTypes = []
    h_criticalMSProjs = []
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
        self.eventrates = [10, 10, 10, 10, 10, 10] #TODO lower eventrates
        self.networkParams = [self.eventskew,self.number_eventtypes,self.node_event_ratio,self.nwSize,min(self.eventrates)/max(self.eventrates)]
        self.primitiveEvents= ['A', 'B', 'C', 'D', 'E', 'F'] #generate_events(self.eventrates,node_event_ratio)
        #self.network = self.network[0].add_child(self.network[1]), self.network[0].add_child(self.network[2]),self.network[1].add_child(self.network[3]), self.network[1].add_child(self.network[4]),self.network[2].add_child(self.network[5]),self.network[2].add_child(self.network[6]),self.network[3].add_child(self.network[7]),self.network[3].add_child(self.network[8]),self.network[4].add_child(self.network[9]),self.network[5].add_child(self.network[10]),self.network[6].add_child(self.network[11]) #create_random_tree(nwSize,self.eventrates,node_event_ratio,max_partens) 
        #root = self.network[0]

        # 1️⃣ Knoten erstellen
        n0 = Node(0, compute_power=np.inf, memory=np.inf)
        n1 = Node(1, compute_power=3, memory=3)
        n2 = Node(2, compute_power=3, memory=3)
        n3 = Node(3, compute_power=1, memory=1)
        n4 = Node(4, compute_power=1, memory=1)
        n5 = Node(5, compute_power=1, memory=1)
        n6 = Node(6, compute_power=1, memory=1)
        n7 = Node(7, compute_power=0, memory=0)
        n8 = Node(8, compute_power=0, memory=0)
        n9 = Node(9, compute_power=0, memory=0)
        n10 = Node(10, compute_power=0, memory=0)
        n11 = Node(11, compute_power=0, memory=0)

        # 2️⃣ Beziehungen aufbauen
        # Beziehungen aufbauen (Child + Parent korrekt)
        n0.Child.append(n1)
        n1.Parent.append(n0)

        n0.Child.append(n2)
        n2.Parent.append(n0)

        n1.Child.append(n3)
        n3.Parent.append(n1)

        n1.Child.append(n4)
        n4.Parent.append(n1)

        n2.Child.append(n5)
        n5.Parent.append(n2)

        n2.Child.append(n6)
        n6.Parent.append(n2)

        n3.Child.append(n7)
        n7.Parent.append(n3)

        n3.Child.append(n8)
        n8.Parent.append(n3)

        n4.Child.append(n9)
        n4.Child.append(n10)
        n9.Parent.append(n4)
        n10.Parent.append(n4)

        n5.Child.append(n10)
        n5.Child.append(n9)
        n9.Parent.append(n5)
        n10.Parent.append(n5)

        n6.Child.append(n11)
        n11.Parent.append(n6)


        
        # 3️⃣ Netzwerkliste definieren
        self.network = [n0, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11]
        root = n0
        # Beispiel: Hierarchie-Level festlegen
        # for node in self.network:
        #     if node.id in [0]:     # Root
        #         node.hierarchy_level = 0
        #     elif node.id in [1, 2]:  # Fog
        #         node.hierarchy_level = 1
        #     elif node.id in [3, 4, 5, 6]:  # Edge
        #         node.hierarchy_level = 2
        #     else:                  # Sensor
        #         node.hierarchy_level = 3

        # 4️⃣ Eventrates für Leaf Nodes setzen
        eventrate_map = {
            7: [10, 10, 10, 0, 0, 0],  # A
            8: [0, 10, 0, 0, 0, 0],  # B
            9: [0, 0, 10, 0, 10, 0], # C,D
            10: [0, 10, 0, 10, 0, 0], # B,D
            11: [0, 0, 0, 0, 10, 10] # E,F
        }
        for node in self.network:
            node.eventrates = eventrate_map.get(node.id, [0] * len(self.eventrates))
        eventrate_map = {
            7: [10, 10, 10, 0, 0, 0],  # A
            8: [0, 10, 0, 0, 0, 0],  # B
            9: [0, 0, 10, 0, 10, 0], # C,D
            10: [0, 10, 0, 10, 0, 0], # B,D
            11: [0, 0, 0, 0, 10, 10] # E,F
        }
        for node in self.network:
            node.eventrates = eventrate_map.get(node.id, [0] * len(self.eventrates))
        #create_random_tree(nwSize,self.eventrates,node_event_ratio,max_partens) #
        self.graph = create_fog_graph(self.network)
        self.allPairs = populate_allPairs(self.graph)
        self.h_longestPath = getLongest(self.allPairs)
        self.query_workload = [SEQ(PrimEvent("A"), PrimEvent("C"),PrimEvent("B")),AND(PrimEvent("F"), PrimEvent("D"), PrimEvent("C"), PrimEvent("B")),
                               AND(PrimEvent("A"), SEQ(PrimEvent("E"), PrimEvent("D"), PrimEvent("C")))]

        #generate_workload(query_size,query_length,self.primitiveEvents)
        self.selectivities,self.selectivitiesExperimentData = initialize_selectivities(self.primitiveEvents)
        self.config_single = generate_config_buffer(self.network,self.query_workload,self.selectivities)
        self.single_selectivity = initializeSingleSelectivity(self.CURRENT_SECTION, self.config_single, self.query_workload)
        #This is important to variious files afterwards
        self.h_network_data,self.h_rates_data,self.h_primEvents,self.h_instances,self.h_nodes = initialize_globals(self.network)
        self.h_eventNodes,self.h_IndexEventNodes = initEventNodes(self.h_nodes,self.h_network_data)
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
    my_ines = INES(12, 0.5, 6, 0.3, 8, 3, 5)
except Exception as e:
    error_message = f"❌ Exception: {str(e)}\n"
    print(error_message)  # Optional: also print to console
