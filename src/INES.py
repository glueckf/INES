from Node import Node
from network import generate_eventrates, create_random_tree,generate_events
from graph import create_fog_graph
from graph import draw_graph
from allPairs import populate_allPairs
from queryworkload import generate_workload

class INES():
    allPais: list
    network: list[Node]
    eventrates: list[list[int]]
    query_workload = None

    primitiveEvents: list[int]
    def __init__(self, nwSize: int, node_event_ratio: float, num_eventtypes: int, eventskew: float, max_partens: int, query_size: int, query_length:int):
        self.eventrates = generate_eventrates(eventskew,num_eventtypes)
        self.primitiveEvents= generate_events(self.eventrates,node_event_ratio)
        root, self.network = create_random_tree(nwSize,self.eventrates,node_event_ratio,max_partens) 
        self.graph = create_fog_graph(self.network)
        self.allPais = populate_allPairs(self.graph)
        self.query_workload = generate_workload(query_size,query_length,self.primitiveEvents)

my_ines = INES(50,0.5,6,0.3,2,3,5)
print(my_ines.query_workload)
#print(my_ines.allPais)
#draw_graph(my_ines.graph)

# for i in my_ines.network:
#     print(i)