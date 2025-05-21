"""
Generate network with given size (nwsize), node-event ratio (node_event_ratio), 
number of event types (num_eventtypes), event rate skew (eventskew)-
"""

import numpy as np
import random
from Node import Node
import pandas as pd


def generate_eventrates(eventskew, numb_eventtypes,total_count=10000):
        # Ensure all eventrates are greater than 0
    min_value = 1
    max_value = 1000
    # Generate ranks
    k = np.arange(1, numb_eventtypes + 1)
    
    # Calculate weights based on Zipf's law
    weights = 1 / k ** eventskew
    weights /= weights.sum()  # Normalize to sum to 1
    
    # Sample event rates from multinomial distribution
    eventrates = np.random.multinomial(total_count, weights)

    # TODO: Frage: Warum total_count = 10000 und clipping bei 1000, wenn alle weights immer größer sind als 0.1?
    
    # Clip event rates to be within min and max values
    eventrates = np.clip(eventrates, min_value, max_value)
    
    return eventrates
    

# At one Node show in Array the Events which are generated at each node
# Looping through all Eventrates 
def generate_events(eventrates, node_event_ratio):
    myevents = []
    for i in range(len(eventrates)):
        x = np.random.uniform(0,1)
        if x < node_event_ratio:
            myevents.append(int(eventrates[i]))
        else:
            myevents.append(0)
    
    return myevents


def create_random_tree(nwsize, eventrates, node_event_ratio, max_parents: int = 1):
    """Create a random tree network with the specified parameters.
    
    Args:
        nwsize: Number of nodes in the network
        eventrates: List of event rates for each event type
        node_event_ratio: Probability of a node generating an event
        max_parents: Maximum number of parents a node can have
        
    Returns:
        Tuple of (root_node, all_nodes_list)
    """
    import math
    if nwsize <= 0:
        return None, []
    
    # Initialize the network nodes list
    nodes = []
    
    # Calculate the number of tree levels based on network size
    levels = math.ceil(math.log2(nwsize))
    
    # Create the root node (has infinite resources)
    root = Node(id=0, compute_power=math.inf, memory=math.inf)
    nodes.append(root)
    
    # Store nodes by level for easier parent selection
    nodes_by_level = {0: [root]}
    
    # Create all remaining nodes
    for node_id in range(1, nwsize):
        # Calculate the target level for this node to create a balanced tree
        level = min(levels - 1, node_id // (nwsize // levels) + 1)
        
        # Assign resources (higher levels have lower resources)
        resources = levels - level
        
        # Create new node
        new_node = Node(id=node_id, compute_power=resources, memory=resources)
        
        # Make sure we have a list for this level
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        
        # Connect to parent nodes from the previous level
        parent_count = random.randint(1, max_parents)
        available_parents = nodes_by_level[level - 1]
        parent_nodes = random.sample(available_parents, min(len(available_parents), parent_count))
        
        # Set up parent-child relationships
        for parent in parent_nodes:
            new_node.Parent.append(parent)
            parent.Child.append(new_node)
        
        # Add new node to the network and its level
        nodes.append(new_node)
        nodes_by_level[level].append(new_node)
    
    # Assign event rates to nodes
    _assign_event_rates(nodes, eventrates, node_event_ratio)
    
    return root, nodes


def _assign_event_rates(nodes, eventrates, node_event_ratio):
    """Assign event rates to nodes in the network.
    
    Leaf nodes get randomly generated event rates, while non-leaf nodes start with
    event rates of 0. Also ensures every event type has at least one producer.
    
    Args:
        nodes: List of all nodes in the network
        eventrates: List of event rates for each event type
        node_event_ratio: Probability of a node generating an event
    """
    # Identify leaf nodes (nodes with no children)
    leaf_nodes = [node for node in nodes if len(node.Child) == 0]
    
    # Set up initial event rates
    for node in nodes:
        if node in leaf_nodes:
            # Leaf nodes generate events
            node.eventrates = generate_events(eventrates, node_event_ratio)
        else:
            # Non-leaf nodes start with zero event rates
            node.eventrates = [0] * len(eventrates)
    
    # Check if any event type has no producer
    event_sums = pd.DataFrame([node.eventrates for node in leaf_nodes])
    
    # For each event type with no producers, assign it to a random leaf node
    for event_idx in event_sums.columns:
        if event_sums[event_idx].sum() == 0:
            random_leaf = random.choice(leaf_nodes)
            random_leaf.eventrates[event_idx] = eventrates[event_idx]
