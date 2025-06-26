"""
Hardcoded INES scenario that reproduces exact results every time.

This module creates the exact network topology, event distribution, and queries
from the successful simulation run to ensure reproducible results for testing
and demonstration purposes.
"""

from ...Node import Node
from ...helper.Tree import AND, SEQ, PrimEvent
import math


def create_hardcoded_network():
    """
    Create the exact 12-node network topology from the simulation logs.
    
    Network structure based on the logs:
    - Node 0: Root with children [1, 2]
    - Node 1: Children [3, 4, 5], Parent [0]  
    - Node 2: Children [3, 4, 5], Parent [0]
    - Node 3: Children [6, 7, 8, 9, 10, 11], Parents [1, 2]
    - Node 4: Children [6, 7, 8, 9, 10, 11], Parents [1, 2]
    - Node 5: Children [6, 7, 8, 9, 10, 11], Parents [2, 1]
    - Nodes 6-11: Leaf nodes with various parents
    
    Event distribution from logs:
    - Node 6: A:1000, B:1000, C:1000, F:1000
    - Node 7: B:1000, C:1000, F:1000
    - Node 8: B:1000, C:1000
    - Node 9: B:1000, F:1000
    - Node 10: A:1000, C:1000, D:1000
    - Node 11: A:1000, C:1000, E:1000
    """
    network = []
    
    # Create all 12 nodes
    for i in range(12):
        if i == 0:
            # Root node with infinite resources
            node = Node(i, math.inf, math.inf)
        else:
            # Other nodes with finite resources
            node = Node(i, 5, 5)
        
        # Initialize empty event rates (6 event types: A, B, C, D, E, F)
        node.eventrates = [0, 0, 0, 0, 0, 0]
        network.append(node)
    
    # Set up the exact topology from the logs
    
    # Node 0 (root): children [1, 2]
    network[0].Child = [network[1], network[2]]
    
    # Node 1: parent [0], children [3, 4, 5]
    network[1].Parent = [network[0]]
    network[1].Child = [network[3], network[4], network[5]]
    
    # Node 2: parent [0], children [3, 4, 5]  
    network[2].Parent = [network[0]]
    network[2].Child = [network[3], network[4], network[5]]
    
    # Node 3: parents [1, 2], children [6, 7, 8, 9, 10, 11]
    network[3].Parent = [network[1], network[2]]
    network[3].Child = [network[6], network[7], network[8], network[9], network[10], network[11]]
    
    # Node 4: parents [1, 2], children [6, 7, 8, 9, 10, 11]
    network[4].Parent = [network[1], network[2]]
    network[4].Child = [network[6], network[7], network[8], network[9], network[10], network[11]]
    
    # Node 5: parents [2, 1], children [6, 7, 8, 9, 10, 11]
    network[5].Parent = [network[2], network[1]]
    network[5].Child = [network[6], network[7], network[8], network[9], network[10], network[11]]
    
    # Leaf nodes 6-11: set up parents according to logs
    # Node 6: parents [5, 3, 4]
    network[6].Parent = [network[5], network[3], network[4]]
    
    # Node 7: parents [5, 3, 4]  
    network[7].Parent = [network[5], network[3], network[4]]
    
    # Node 8: parents [4, 3, 5]
    network[8].Parent = [network[4], network[3], network[5]]
    
    # Node 9: parents [3, 5, 4]
    network[9].Parent = [network[3], network[5], network[4]]
    
    # Node 10: parents [4, 3, 5]
    network[10].Parent = [network[4], network[3], network[5]]
    
    # Node 11: parents [3, 5, 4]
    network[11].Parent = [network[3], network[5], network[4]]
    
    # Set the exact event distribution from the logs
    # Event mapping: A=0, B=1, C=2, D=3, E=4, F=5
    
    # Node 6: A:1000, B:1000, C:1000, F:1000
    network[6].eventrates = [1000, 1000, 1000, 0, 0, 1000]
    
    # Node 7: B:1000, C:1000, F:1000
    network[7].eventrates = [0, 1000, 1000, 0, 0, 1000]
    
    # Node 8: B:1000, C:1000
    network[8].eventrates = [0, 1000, 1000, 0, 0, 0]
    
    # Node 9: B:1000, F:1000
    network[9].eventrates = [0, 1000, 0, 0, 0, 1000]
    
    # Node 10: A:1000, C:1000, D:1000
    network[10].eventrates = [1000, 0, 1000, 1000, 0, 0]
    
    # Node 11: A:1000, C:1000, E:1000
    network[11].eventrates = [1000, 0, 1000, 0, 1000, 0]
    
    return network[0], network


def create_hardcoded_queries():
    """
    Create the exact queries from the simulation logs:
    - Query 1: AND(A, B, E) 
    - Query 2: SEQ(F, C, A)
    - Query 3: SEQ(F, D, B)
    
    Creates queries using the same method as the original queryworkload.py
    including the number_children post-processing step.
    """
    from queryworkload import number_children
    
    queries = []
    
    # Create primitive events
    event_A = PrimEvent('A')
    event_B = PrimEvent('B')
    event_C = PrimEvent('C')
    event_D = PrimEvent('D')
    event_E = PrimEvent('E')
    event_F = PrimEvent('F')
    
    # Query 1: AND(A, B, E) (events: ['A', 'B', 'E'])
    # Create empty AND query and assign children like the original code
    and_query = AND()
    and_query.children = [event_A, event_B, event_E]
    and_query = number_children(and_query)  # Apply same post-processing as original
    queries.append(and_query)
    
    # Query 2: SEQ(F, C, A) (events: ['F', 'C', 'A'])
    seq_query1 = SEQ()
    seq_query1.children = [event_F, event_C, event_A]
    seq_query1 = number_children(seq_query1)  # Apply same post-processing as original
    queries.append(seq_query1)
    
    # Query 3: SEQ(F, D, B) (events: ['F', 'D', 'B'])
    seq_query2 = SEQ()
    seq_query2.children = [event_F, event_D, event_B]
    seq_query2 = number_children(seq_query2)  # Apply same post-processing as original
    queries.append(seq_query2)
    
    return queries


def get_hardcoded_eventrates():
    """
    Return the exact event rates from the logs.
    From logs: "Generated event rates for 6 event types: [1000 1000 1000 1000 1000 1000]"
    """
    return [1000, 1000, 1000, 1000, 1000, 1000]


def get_hardcoded_primitive_events():
    """
    Return the exact primitive events from the logs.
    From logs: "Generated 6 primitive events: [1000, 1000, 0, 0, 0, 0]"
    """
    return [1000, 1000, 0, 0, 0, 0]


def get_hardcoded_parameters():
    """
    Return the exact simulation parameters that produced the logged results.
    """
    return {
        'nodes': 12,
        'node_event_ratio': 0.5,
        'num_eventtypes': 6,
        'eventskew': 0.3,
        'max_parents': 10,
        'query_size': 3,
        'query_length': 5
    }


def get_expected_results():
    """
    Return the expected results from the successful simulation run.
    """
    return {
        'transmission_ratio': 0.31380392156862746,
        'total_cost': 51000.0,
        'optimized_cost': 16004.0,
        'savings_percentage': 68.6,
        'longest_path': 1.6666666666666667
    }


def print_hardcoded_scenario_info():
    """
    Print information about the hardcoded scenario for verification.
    """
    print("="*80)
    print("HARDCODED SCENARIO INFORMATION")
    print("="*80)
    print("This scenario reproduces the exact network topology and queries from")
    print("the successful simulation run logged on 2025-06-25 14:54:51")
    print()
    print("Network Topology:")
    print("- 12 nodes with specific parent-child relationships")
    print("- Event distribution exactly matching the logs")
    print("- Root node (0) with infinite resources")
    print()
    print("Queries:")
    print("- Query 1: AND(A, B, E)")
    print("- Query 2: SEQ(F, C, A)")  
    print("- Query 3: SEQ(F, D, B)")
    print()
    print("Expected Results:")
    results = get_expected_results()
    print(f"- Transmission Ratio: {results['transmission_ratio']:.3f}")
    print(f"- Total Cost: {results['total_cost']:.0f}")
    print(f"- Optimized Cost: {results['optimized_cost']:.0f}")
    print(f"- Savings: {results['savings_percentage']:.1f}%")
    print("="*80)


if __name__ == "__main__":
    # Test the hardcoded scenario
    print_hardcoded_scenario_info()
    
    # Create and verify the network
    root, network = create_hardcoded_network()
    print(f"\nCreated network with {len(network)} nodes")
    
    # Create and verify the queries
    queries = create_hardcoded_queries()
    print(f"Created {len(queries)} queries")
    
    print("\nHardcoded scenario ready for use!")