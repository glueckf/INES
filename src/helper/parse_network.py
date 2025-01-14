import string

# Private global variables
_network = None
_rates = None
_primEvents = None
_instances = None
_nodes = None

# Function to initialize global variables
def initialize_globals(network_data):
    global _network, _rates, _primEvents, _instances, _nodes
    
    # Initialize global variables
    _network = {}
    _rates = {}
    _primEvents = []
    _instances = {}
    _nodes = {}

    ind = 0
    for node in network_data:
        _network[ind] = []
        for eventtype in range(len(node.eventrates)):
            if node.eventrates[eventtype] > 0:
                event = string.ascii_uppercase[eventtype]
                _network[ind].append(event)
                if event not in _primEvents:
                    _primEvents.append(event)
                if event not in _rates:
                    _rates[event] = float(node.eventrates[eventtype])
        ind += 1

    # Populate nodes dictionary
    for i in _network.keys():
        for event in _network[i]:
            if event not in _nodes:
                _nodes[event] = [i]
            else:
                _nodes[event].append(i)

    # Populate instances dictionary
    for i in _primEvents:
        _instances[i] = len(_nodes[i])
    return _network,_rates,_primEvents,_instances,_nodes
# Getter functions with lazy initialization
def get_network(network_data):
    if _network is None:
        initialize_globals(network_data)
    return _network

def get_rates(network_data):
    if _rates is None:
        initialize_globals(network_data)
    return _rates

def get_primEvents(network_data):
    if _primEvents is None:
        initialize_globals(network_data)
    return _primEvents

def get_instances(network_data):
    if _instances is None:
        initialize_globals(network_data)
    return _instances

def get_nodes(network_data):
    if _nodes is None:
        initialize_globals(network_data)
    return _nodes

# Utility function for accessing network events
def events(node, network_data):
    if _network is None:
        initialize_globals(network_data)
    return _network[node]

# Function to calculate instances from projections
def instances_func(proj, network_data):    
    if _nodes is None:
        initialize_globals(network_data)
    
    num = 1
    for i in proj:
        if len(i) == 1:
            num *= len(_nodes[i])
        else:
            for j in i:
                num *= len(_nodes[j])
    return num
