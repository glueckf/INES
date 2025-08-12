"""
Subgraph extraction and remapping utilities.

This module handles the extraction of relevant subgraphs for placement
computation and provides utilities for node remapping.
"""

from typing import Dict, List, Any, Set
import networkx as nx
from helper.structures import getNodes
from Node import Node
from .logging import get_placement_logger
from .state import SubgraphBundle

logger = get_placement_logger(__name__)


def extract_subgraph(placement_node: int, network: List[Node], graph: nx.Graph, 
                    combination: List[Any], index_event_nodes: Dict[str, List[Any]], 
                    event_nodes: List[List[Any]], routing_dict: Dict[int, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Extract a subgraph for a given placement node containing all relevant nodes and paths
    needed for routing event data to that node.
    
    This function performs the following key operations:
    1. **Relevant node collection** - Identifies all nodes that need to be included
    2. **Subgraph creation** - Extracts connected subgraph from the main graph
    3. **Node ID remapping** - Creates sequential IDs for the subgraph (0, 1, 2, ...)
    4. **Data structure adaptation** - Creates subgraph versions of all required matrices
    
    Args:
        placement_node: The target node where the projection will be placed
        network: List of Node objects representing the network topology
        graph: NetworkX graph representing the network connectivity
        combination: List of event types that need to be routed to the placement node
        index_event_nodes: Dictionary mapping event types to their ETB instances
        event_nodes: Matrix mapping event types to nodes
        routing_dict: Dictionary containing routing information including shortest paths
        
    Returns:
        dict: A dictionary containing the subgraph data structures needed for generate_prePP:
            - 'subgraph': NetworkX subgraph containing only relevant nodes and edges
            - 'node_mapping': Mapping from original node IDs to subgraph node IDs
            - 'reverse_mapping': Mapping from subgraph node IDs to original node IDs
            - 'event_nodes_sub': Filtered event nodes matrix for the subgraph
            - 'index_event_nodes_sub': Filtered index event nodes for the subgraph
            - 'network_data_sub': Network data filtered for subgraph nodes
            - 'all_pairs_sub': Distance matrix for subgraph nodes
            - 'relevant_nodes': Set of all nodes included in the subgraph
            - 'placement_node_remapped': Remapped placement node ID
            - 'sub_network': List of Node objects for the subgraph

    Comments:
        As of now, this function implements a basic extraction of relevant nodes and paths like in original.
        In the future this could be improved by adding simpler filter techniques and existing structures.
        TODO: Improve extraction logic.
    """
    logger.debug(f"Extracting subgraph for placement node {placement_node}")
    
    # Start with the placement node
    relevant_nodes = {placement_node}
    
    # Find all source nodes for events in the combination
    for event_type in combination:
        if event_type in index_event_nodes:
            for etb in index_event_nodes[event_type]:
                # Get all nodes that produce this ETB
                source_nodes = getNodes(etb, event_nodes, index_event_nodes)
                relevant_nodes.update(source_nodes)
                
                # Add all nodes on the shortest paths from sources to placement node
                for source_node in source_nodes:
                    if source_node != placement_node:
                        path_nodes = _get_path_nodes(source_node, placement_node, routing_dict, graph)
                        relevant_nodes.update(path_nodes)
    
    logger.debug(f"Collected {len(relevant_nodes)} relevant nodes: {sorted(list(relevant_nodes))}")
    
    # Create node mappings and subgraph structures
    node_mapping, reverse_mapping = _create_node_mappings(relevant_nodes)
    subgraph_data = _build_subgraph_structures(
        relevant_nodes, node_mapping, reverse_mapping, graph, network, 
        event_nodes, index_event_nodes, placement_node
    )
    
    logger.info(f"Subgraph extracted: {len(relevant_nodes)} nodes, {subgraph_data['subgraph'].number_of_edges()} edges")
    
    return subgraph_data


def _get_path_nodes(source_node: int, placement_node: int, routing_dict: Dict[int, Dict[int, Dict[str, Any]]], 
                   graph: nx.Graph) -> Set[int]:
    """
    Get all nodes on the path from source to placement node.
    
    Args:
        source_node: Source node ID
        placement_node: Placement node ID
        routing_dict: Routing dictionary with path information
        graph: NetworkX graph for fallback path computation
        
    Returns:
        Set of node IDs on the path
    """
    try:
        # Try to get path from routing dict
        if placement_node in routing_dict and source_node in routing_dict[placement_node]:
            path_info = routing_dict[placement_node][source_node]
            if 'path' in path_info:
                return set(path_info['path'])
            elif 'shortest_path' in path_info:
                return set(path_info['shortest_path'])
        
        # Fallback: compute shortest path
        path_nodes = nx.shortest_path(graph, source_node, placement_node)
        return set(path_nodes)
        
    except (nx.NetworkXNoPath, KeyError):
        # If no path exists, just return the source node
        logger.warning(f"No path found from {source_node} to {placement_node}")
        return {source_node}


def _create_node_mappings(relevant_nodes: Set[int]) -> tuple[Dict[int, int], Dict[int, int]]:
    """
    Create bidirectional mappings between original and subgraph node IDs.
    
    Args:
        relevant_nodes: Set of relevant node IDs from original graph
        
    Returns:
        Tuple of (node_mapping, reverse_mapping) dictionaries
    """
    relevant_nodes_list = sorted(list(relevant_nodes))
    node_mapping = {orig_id: new_id for new_id, orig_id in enumerate(relevant_nodes_list)}
    reverse_mapping = {new_id: orig_id for orig_id, new_id in node_mapping.items()}
    
    logger.debug(f"Created node mappings: {len(node_mapping)} nodes remapped")
    return node_mapping, reverse_mapping


def _build_subgraph_structures(relevant_nodes: Set[int], node_mapping: Dict[int, int], 
                             reverse_mapping: Dict[int, int], graph: nx.Graph, network: List[Node],
                             event_nodes: List[List[Any]], index_event_nodes: Dict[str, List[Any]], 
                             placement_node: int) -> Dict[str, Any]:
    """
    Build all the data structures needed for the subgraph.
    
    Args:
        relevant_nodes: Set of relevant node IDs
        node_mapping: Original to subgraph ID mapping
        reverse_mapping: Subgraph to original ID mapping
        graph: Original NetworkX graph
        network: Original network node list
        event_nodes: Original event nodes matrix
        index_event_nodes: Original index event nodes
        placement_node: Original placement node ID
        
    Returns:
        Dictionary with all subgraph structures
    """
    # Create remapped NetworkX subgraph
    remapped_subgraph = _create_remapped_subgraph(relevant_nodes, node_mapping, graph)
    
    # Create filtered event structures
    event_nodes_sub = _create_subgraph_event_nodes(relevant_nodes, event_nodes)
    index_event_nodes_sub = index_event_nodes.copy()  # ETB structure stays the same
    
    # Create network data for subgraph
    network_data_sub = _create_subgraph_network_data(reverse_mapping, network)
    
    # Create all-pairs distance matrix
    all_pairs_sub = _create_subgraph_distance_matrix(remapped_subgraph)
    
    # Create sub-network node objects
    sub_network = _create_subgraph_network_objects(reverse_mapping, network, node_mapping)
    
    return {
        'subgraph': remapped_subgraph,
        'node_mapping': node_mapping,
        'reverse_mapping': reverse_mapping,
        'event_nodes_sub': event_nodes_sub,
        'index_event_nodes_sub': index_event_nodes_sub,
        'network_data_sub': network_data_sub,
        'all_pairs_sub': all_pairs_sub,
        'relevant_nodes': relevant_nodes,
        'placement_node_remapped': node_mapping[placement_node],
        'sub_network': sub_network
    }


def _create_remapped_subgraph(relevant_nodes: Set[int], node_mapping: Dict[int, int], graph: nx.Graph) -> nx.Graph:
    """Create a NetworkX subgraph with remapped sequential node IDs."""
    remapped_subgraph = nx.Graph()
    relevant_nodes_list = sorted(list(relevant_nodes))
    
    # Add nodes with remapped IDs
    for orig_node in relevant_nodes_list:
        new_node_id = node_mapping[orig_node]
        remapped_subgraph.add_node(new_node_id)
        
        # Copy node attributes if they exist
        if graph.has_node(orig_node):
            for attr_key, attr_val in graph.nodes[orig_node].items():
                remapped_subgraph.nodes[new_node_id][attr_key] = attr_val
    
    # Add edges with remapped node IDs
    subgraph = graph.subgraph(relevant_nodes)
    for orig_u, orig_v in subgraph.edges():
        new_u = node_mapping[orig_u]
        new_v = node_mapping[orig_v]
        remapped_subgraph.add_edge(new_u, new_v)
        
        # Copy edge attributes if they exist
        if graph.has_edge(orig_u, orig_v):
            for attr_key, attr_val in graph.edges[orig_u, orig_v].items():
                remapped_subgraph.edges[new_u, new_v][attr_key] = attr_val
    
    return remapped_subgraph


def _create_subgraph_event_nodes(relevant_nodes: Set[int], event_nodes: List[List[Any]]) -> List[List[Any]]:
    """Create filtered event nodes matrix for subgraph."""
    relevant_nodes_list = sorted(list(relevant_nodes))
    event_nodes_sub = []
    
    for event_row in event_nodes:
        # Create new row with only relevant nodes
        new_row = [event_row[orig_id] if orig_id < len(event_row) else 0
                   for orig_id in relevant_nodes_list]
        event_nodes_sub.append(new_row)
    
    return event_nodes_sub


def _create_subgraph_network_data(reverse_mapping: Dict[int, int], network: List[Node]) -> Dict[int, List[str]]:
    """Create network data mapping for subgraph nodes."""
    network_data_sub = {}
    
    for new_node_id, orig_node_id in reverse_mapping.items():
        # Initialize with empty list (non-leaf nodes don't produce events)
        network_data_sub[new_node_id] = []
        
        # Check if this node produces any events
        if orig_node_id < len(network):
            node_obj = network[orig_node_id]
            if hasattr(node_obj, 'eventrates') and node_obj.eventrates:
                # Find which event types this node produces
                produced_events = []
                for event_idx, rate in enumerate(node_obj.eventrates):
                    if rate > 0:
                        # Convert event index to event type letter
                        event_type = chr(ord('A') + event_idx)
                        produced_events.append(event_type)
                network_data_sub[new_node_id] = produced_events
    
    return network_data_sub


def _create_subgraph_distance_matrix(remapped_subgraph: nx.Graph) -> List[List[float]]:
    """Create all-pairs shortest path matrix for subgraph."""
    try:
        # Compute shortest paths for remapped subgraph
        all_pairs_dict = dict(nx.all_pairs_shortest_path_length(remapped_subgraph))
        
        # Convert to matrix format
        num_nodes = len(remapped_subgraph.nodes)
        all_pairs_sub = [[float('inf')] * num_nodes for _ in range(num_nodes)]
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    all_pairs_sub[i][j] = 0
                elif j in all_pairs_dict.get(i, {}):
                    all_pairs_sub[i][j] = all_pairs_dict[i][j]
        
        return all_pairs_sub
        
    except nx.NetworkXError:
        # Fallback: create basic distance matrix
        num_nodes = len(remapped_subgraph.nodes)
        return [[1 if i != j else 0 for j in range(num_nodes)] for i in range(num_nodes)]


def _create_subgraph_network_objects(reverse_mapping: Dict[int, int], network: List[Node], 
                                   node_mapping: Dict[int, int]) -> List[Node]:
    """Create Node objects for the subgraph with remapped relationships."""
    sub_network = []
    
    for new_node_id in sorted(reverse_mapping.keys()):
        orig_node_id = reverse_mapping[new_node_id]
        
        # Create new Node object for subgraph
        if orig_node_id < len(network):
            orig_node = network[orig_node_id]
            # Create new node with remapped ID but original attributes
            sub_node = Node(new_node_id, orig_node.computational_power, orig_node.memory)
            sub_node.eventrates = orig_node.eventrates.copy() if orig_node.eventrates else []
            
            # Initialize empty relationships (will be populated later if needed)
            sub_node.Parent = []
            sub_node.Child = []
            sub_node.Sibling = []
            
            sub_network.append(sub_node)
    
    return sub_network