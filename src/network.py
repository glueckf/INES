"""
Generate network with given size (nwsize), node-event ratio (node_event_ratio), 
number of event types (num_eventtypes), event rate skew (eventskew)-
"""

import pickle
import numpy as np
import random
from Node import Node
import argparse
import pandas as pd

#sys.setrecursionlimit(200000)


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
    
    # Clip event rates to be within min and max values
    eventrates = np.clip(eventrates, min_value, max_value)
    
    return eventrates
    
    # eventrates[eventrates < 1] = 1
    # return eventrates

# At one Node show in Array the Events which are generated at each node
# Looping through all Eventrates 
def generate_events(eventrates, n_e_r):
    myevents = []
    for i in range(len(eventrates)):
        x = np.random.uniform(0,1)
        if x < n_e_r:
            myevents.append(int(eventrates[i]))
        else:
            myevents.append(0)
    
    return myevents


def create_random_tree(nwsize, eventrates, node_event_ratio, max_parents: int = 1):
        if nwsize <= 0:
            return None
        import math
        # Initialize the list to store all nodes
        nw = []

        levels = math.ceil(math.log2(nwsize))
        print(levels)
        # Create the root node
        root = Node(id=0, compute_power=math.inf, memory=math.inf )#, eventrate=generate_events(eventrates, node_event_ratio))
        nw.append(root)

        # Track nodes by level to manage the structure and prevent imbalance
        level_nodes = {0: [root]}

        # Create remaining nodes and build the tree
        for node_id in range(1, nwsize):
            # Determine the level for the new node
            level = min(levels - 1, node_id // (nwsize // levels) + 1)
            
            # Compute power and memory decrease as the level increases
            compute_power = levels - level
            memore = levels - level

            # Create the new node
            new_node = Node(id=node_id, compute_power=compute_power, memory=memore)

            # Ensure level-specific nodes exist in the dictionary
            if level not in level_nodes:
                level_nodes[level] = []
            
            # Randomly choose the number of parents between 1 and max_parents
            num_parents = random.randint(1,max_parents)
                
            # Randomly choose parents from the previous level
            parent_nodes = random.sample(level_nodes[level - 1], min(len(level_nodes[level - 1]), num_parents))

            # Set parents and add the new node to each parent's list of children
            for parent_node in parent_nodes:
                new_node.Parent.append(parent_node)
                parent_node.Child.append(new_node)


            # Add new node to the list and to the level-specific tracking
            nw.append(new_node)
            level_nodes[level].append(new_node)

        # Assign event rates to leaf nodes and initialize non-leaf nodes with empty event rates
        for node in nw:
            if len(node.Child) == 0:
   
                evtrate = generate_events(eventrates, node_event_ratio)

                # with open('PrimitiveEvents', 'wb') as f:
                #     pickle.dump(evtrate, f)
                node.eventrates = evtrate
            else:
                node.eventrates = [0] * len(eventrates)
                
        eventrates_df = pd.DataFrame([node.eventrates for node in nw if len(node.Child) == 0])
          # Check if any column (representing an event) has only 0
        for column in eventrates_df.columns:
            if eventrates_df[column].sum() == 0:
                # Select a random leaf node and assign the eventrate from eventrates array
                # print(f"Assigning eventrate to a random leaf node {column}")
                random_leaf_node = random.choice([node for node in nw if len(node.Child) == 0])
                #print(random_leaf_node)
                random_leaf_node.eventrates[column] = eventrates[column]
        print(eventrates)
        for column in eventrates_df.columns:
            if eventrates_df[column].sum() == 0:
                print("Still 0 ")
        # post_order_sum_events(root)
        return root, nw

def parse_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Simulation parameters")

    # Add arguments with default values
    parser.add_argument('--nwsize', '-nw', type=int, default=10, help='Network size (default: 10)')
    parser.add_argument('--node_event_ratio', '-ner', type=float, default=0.5, help='Node event ratio (default: 0.5)')
    parser.add_argument('--num_eventtypes', '-ne', type=int, default=6, help='Number of event types (default: 20)')
    parser.add_argument('--eventskew', '-es', type=float, default=1.3, help='Event skew (default: 1.3)')
    parser.add_argument('--swaps', '-sw', type=int, default=0, help='Number of swaps (default: 0)')
    parser.add_argument('--toFile', '-tf', action='store_true', help='Write event types to file')
    parser.add_argument('--max_parents', '-mp', type=int, default=1, help='Maximum number of parents per node (default: 1)')
    parser.add_argument('--eventtype', '-et', type=str, default=None, help='Event type for experiments')
    parser.add_argument('--param', '-p', type=str, default=None, help='Parameter for event type (max/min) in experiments')

    # Parse the arguments
    args = parser.parse_args()

    # Return parsed arguments as a dictionary for convenience
    return vars(args)

def main():
        # Parse command-line arguments
    args = parse_arguments()

    # Getting access to the Arguments
    nwsize = args['nwsize']
    node_event_ratio = args['node_event_ratio']
    num_eventtypes = args['num_eventtypes']
    eventskew = args['eventskew']
    toFile = args['toFile']
    swaps = args['swaps']
    max_parents = args['max_parents']
    eventtype = args['eventtype']
    param = args['param']

    # The rest of your simulation logic goes here
    print(f"Max parents for nodes: {max_parents}")
    print(f"Network size: {nwsize}, Node event ratio: {node_event_ratio}, Event skew: {eventskew}")
    print(f"Number of event types: {num_eventtypes}, Swaps: {swaps}, To file: {toFile}")
    print(f"Event type: {eventtype}, Param: {param}")

    #eventrates = sorted(generate_eventrates(eventskew,num_eventtypes))
    eventrates =  generate_eventrates(eventskew,num_eventtypes)
    
    print(f"Eventrates: {eventrates}")
    
    
    
    nw = []
    root, nw = create_random_tree(nwsize, eventrates, node_event_ratio, max_parents)
    # for node in range(nwsize):
    #     no = Node(node, 0, 0, generate_events(eventrates, node_event_ratio))
    #     nw.append(no)
        
    #print(nw)     
    
    """
    TODO Rebuild the check for allEvents again
    """
    # print(allEvents(nw))
    # while not allEvents(nw):
    #     nw = []    

    #     for node in range(nwsize):
    #         nw.append(generate_events(eventrates, node_event_ratio))


    ## INSERT NETWORK HERE
    #nw = [[2970, 2000, 322, 600, 960, 458, 2, 2, 40],[2970, 2000, 322, 600, 960, 458, 2, 2, 40],[2970, 2000, 322, 600, 960, 458, 2, 2, 40],[2970, 2000, 322, 600, 960, 458, 2, 2, 40],[2970, 2000, 322, 600, 960, 458, 2, 2, 40],[2970, 2000, 322, 600, 960, 458, 2, 2, 40],[2970, 2000, 322, 600, 960, 458, 2, 2, 40],[2970, 2000, 322, 600, 960, 458, 2, 2, 40],[2970, 2000, 322, 600, 960, 458, 2, 2, 40],[2970, 2000, 322, 600, 960, 458, 2, 2, 40]]

    networkExperimentData = [eventskew, num_eventtypes, node_event_ratio, nwsize, min(eventrates)/max(eventrates)]
    with open('networkExperimentData', 'wb') as networkExperimentDataFile:
        pickle.dump(networkExperimentData, networkExperimentDataFile)
    
    with open('network', 'wb') as network_file:
          pickle.dump(nw, network_file)      
          
         
    
   
    print("NETWORK")  
    print("--------") 
    for i in range(len(nw)):
        print(nw[i])
    print("\n")
    
    # nw[0].visualize_tree(nw[0])
    
    
        
# if __name__ == "__main__":
#     main()


        



