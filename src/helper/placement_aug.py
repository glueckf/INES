#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:16:11 2021

@author: samira
"""
import multiprocessing
#from helper.processCombination_aug import *
from helper.filter import getMaximalFilter,getDecomposedTotal
from helper.structures import getNodes,NumETBsByKey,setEventNodes,SiSManageETBs
from projections import returnPartitioning
from functools import partial
from allPairs import find_shortest_path_or_ancestor
import copy
from EvaluationPlan import Instance,Projection
import numpy as np
from helper.filter import getKeySingleSelect
import networkx as nx
from helper.structures import MSManageETBs, getETBs
from networkx.algorithms.approximation import steiner_tree

def getFilters(self, projection, partType): # move to filter file eventually 
        IndexEventNodes = self.h_IndexEventNodes
        projrates = self.h_projrates
        eventNodes = self.h_eventNodes
        totalETBs = 0
        for etb in IndexEventNodes[partType]: #for each multi-sink
            
                numETBs = 1
                node = getNodes(etb, eventNodes, IndexEventNodes)[0]        
                myETBs = getETBs(node, eventNodes,IndexEventNodes)       
                if not set(IndexEventNodes[projection]).issubset(set(getETBs(node, eventNodes, IndexEventNodes))): # it is  checked if the node already received all etbs of the projection, if this is the case its not necessary to reduce something here     
                    # jedes etb eines leaftypes von projection, aufsummieren, wenn von jedem mindestens 1 dann aufmultiplizieren und somit etbs ausrechnen und dann rate der etbs aufsummieren pro knoten
                    for primEvent in projection.leafs():
                        numETBs *= len(list(set(myETBs) & set(IndexEventNodes[primEvent])))
                       
                    totalETBs += numETBs
      #  print("AUTO FILTERS: " + str(totalETBs * projrates[projection][1]))    # if the projection is also input to another projection in the combination, it may also be the case that the nodes of the parttypes already received all instances of the projections, hence filters can't help anymore...
        return totalETBs * projrates[projection][1]

def computeMSplacementCosts(self, projection, combination, partType, sharedDict, noFilter, G):
    #from allPairs import find_shortest_path_or_ancestor
    #import networkx as nx
    
    projFilterDict = self.h_projFilterDict
    IndexEventNodes = self.h_IndexEventNodes
    nodes = self.h_nodes
    eventNodes = self.h_eventNodes

    projrates = self.h_projrates
    mycombi = self.h_mycombi
    singleSelectivities = self.single_selectivity
    rates = self.h_rates_data
    if not hasattr(self, "assigned_queries_per_node"):
        self.assigned_queries_per_node = {}

    costs = 0
    Filters  = []
    
    ##### FILTERS, append maximal filters
    intercombi = []
    
    automaticFilters = 0
    for proj in combination:
        if len(proj) > 1 and len(IndexEventNodes[proj]) == 1:     #here a node can only have already ALL events if the node is a sink for the projection, this case is however already covered in normal placement cost calculation
            automaticFilters +=  getFilters(proj, partType[0]) # TODO first version
            
        intercombi.append(proj)
        if len(proj) > 1 and len(getMaximalFilter(projFilterDict, proj, noFilter)) > 0: # those are the extra events that need to be sent around due to filters
            Filters.append((proj,getMaximalFilter(projFilterDict, proj, noFilter) ))
            #print("Using Filter: " + str(getMaximalFilter(projFilterDict, proj)) + ": " + str(projFilterDict[proj][getMaximalFilter(projFilterDict, proj)][0]) + " instead of " + str(projrates[proj])  )
            for etype in getMaximalFilter(projFilterDict, proj, noFilter):
                intercombi.append(etype)
    combination = list(set(intercombi))
    myPathLength = 0
  
    totalInstances = [] #!
    myNodes = []
    
    for eventtype in mycombi.get(projection, []):
        if eventtype not in IndexEventNodes:
            continue
        for etb in IndexEventNodes[eventtype]:
            candidates = getNodes(etb, eventNodes, IndexEventNodes)
            for node in candidates:
                if self.network[node].computational_power >= projection.computing_requirements:
                    myNodes.append(node)

    # Remove duplicates
    myNodes = list(set(myNodes))

    # Fallback if no valid nodes found
    if not myNodes:
        print(f"[Warning] No valid nodes found for projection {projection}, using fallback node 0")
        myNodes = [0]

        
    myProjection = Projection(projection, {}, myNodes, [], Filters) #!
    for myInput in combination:
            if not partType[0] == myInput:
                    if myInput in sharedDict.keys():                         
                        #result = NEWcomputeMSplacementCosts_Path(projection, [myInput], sharedDict[myInput], noFilter)
                        result = NEWcomputeMSplacementCosts(self, projection, [myInput], sharedDict[myInput], noFilter,G)
                        costs +=  result[0] #fix SharedDict with Filter Inputs
                        
                    else:
                        #result = NEWcomputeMSplacementCosts_Path(projection, [myInput],  partType[0], noFilter) 
                        result = NEWcomputeMSplacementCosts(self, projection, [myInput],  partType[0], noFilter, G) 

                        costs +=  result[0]
                    if result[1] > myPathLength:
                        myPathLength = result[1]
                    
                    myProjection.addInstances(myInput, result[2]) #!
                    totalInstances += result[2] #!
            else:
                myInstances = [Instance(partType[0], partType[0], nodes[partType[0]], {})]
                myProjection.addInstances(partType[0], myInstances)
                    
    # here generate an instance of etbs per parttype and add one line per instance
    MSManageETBs(self, projection, partType[0]) 
       
    
    spawnedInstances = IndexEventNodes[projection]
    myProjection.addSpawned(spawnedInstances)
    # myProjection.sinks = []
    # for sink in myNodes:
    #     if self.network[sink].computational_power >= projection.computing_requirements:
    #         myProjection.sinks.append(sink)
    #         #self.network[sink].computational_power -= projection.computing_requirements
            
    #         # if G is not None and G.has_node(sink):
    #         #     G.nodes[sink]['relevant'] = True
    #         print(f"[Placement] Projection {projection} assigned to Node {sink}, remaining power: {self.network[sink].computational_power}")

    
    
    costs -= automaticFilters            
    
    return costs, myPathLength, myProjection, totalInstances, Filters




def NEWcomputeMSplacementCosts(self, projection, sourcetypes, destinationtypes, noFilter, G):
    from allPairs import create_routing_dict
    routingDict = create_routing_dict(G)
    routingAlgo = dict(nx.all_pairs_shortest_path(G))
    allPairs = self.allPairs

    costs = 0
    longestPath = 0
    newInstances = []

    projFilterDict = self.h_projFilterDict
    IndexEventNodes = self.h_IndexEventNodes
    projrates = self.h_projrates
    singleSelectivities = self.single_selectivity
    mycombi = self.h_mycombi
    rates = self.h_rates_data
    placementTreeDict = self.h_placementTreeDict
    eventNodes = self.h_eventNodes
    routingInfo = []
    

    # Check filter
    etype = sourcetypes[0]
    with open("msFilter.txt", "w") as f:
        if etype in projFilterDict and getMaximalFilter(projFilterDict, etype, noFilter):
            f.write("VAR=true")
        else:
            f.write("VAR=false")

    # search valid sinks
    destinationtypes = [node for node, neighbors in self.h_network_data.items() if not neighbors and self.network[node].computational_power >= projection.computing_requirements]

    # Ensure hashable type for later usage in placementTreeDict
    destinationtypes_hashed = tuple(destinationtypes)

    # Filters all valid destinationNodes
    destinationNodes = []
    for destination in destinationtypes:
        skip = False    # Flag to determine if we should skip this destination
        for etype in mycombi.get(projection,[]):
            # if eventtype not in IndexEventNodes:
            #     continue
            for etb in IndexEventNodes.get(etype,[]):
                for source in getNodes(etb, eventNodes, IndexEventNodes):
                    # Use the routing_dict to get the common ancestor
                    common_ancestor = routingDict[destination][source]['common_ancestor']
                    if common_ancestor != destination:
                        skip = True
                        break
                if skip: 
                    break  # Break out of the etb loop
            if skip: 
                break  # Break out of the eventtype loop
        if not skip:
            destinationNodes.append(destination)
            if skip:
                break

    # Main logic: for all event types that belong to the projection
    # for etype in mycombi.get(projection,[]):
        if etype not in IndexEventNodes:  # Out of calculation if not relevant
            continue
    for etb in IndexEventNodes.get(etype,[]):
                newInstance = False
                currentSources = getNodes(etb, eventNodes, IndexEventNodes)
                MydestinationNodes = list(set(destinationNodes) - set(currentSources))

                if MydestinationNodes:     
                        for dest in MydestinationNodes:
                            if not dest in getNodes(etb, eventNodes, IndexEventNodes): 
                                mySource = currentSources[0]
                            for source in currentSources:
                                if allPairs[dest][source] < allPairs[dest][mySource]:
                                    mySource = source

                                #node = findBestSource(self,mySource,dest)
                                shortestPath = find_shortest_path_or_ancestor(routingAlgo, mySource, dest)
                                if not shortestPath or not isinstance(shortestPath, list):
                                    print(f"[Warning] No edge from {mySource} to {dest} for etb {etb}")
                                    continue

                                edges = list(zip(shortestPath[:-1], shortestPath[1:]))
                                
                                #print(f"[Tracking] send {etb} from {mySource} to {dest} over {shortestPath}")

                                # Cost calculation
                                if etype in projFilterDict.keys() and  getMaximalFilter(projFilterDict, etype, noFilter): #case input projection has filter
                                    mycosts =  len(edges) * getDecomposedTotal(getMaximalFilter(projFilterDict, etype, noFilter), type)                    
                                    if len(IndexEventNodes[etype]) > 1 : # filtered projection has ms placement
                                                partType = returnPartitioning(etype, mycombi[etype])[0]                     
                                                mycosts -= len(edges)  * rates[partType] * singleSelectivities[getKeySingleSelect(partType, etype)] * len(IndexEventNodes[etype])
                                                mycosts += len(edges)  * rates[partType] * singleSelectivities[getKeySingleSelect(partType, etype)] 
                                elif len(etype) == 1:
                                    mycosts = len(edges) * rates[etype]
                                else:                                             
                                    num = NumETBsByKey(etb, etype, IndexEventNodes)
                                    mycosts = len(edges) *  projrates[etype][1] * num      # FILTER   

                        # pathlength and costs
                        costs += mycosts
                        longestPath = max(longestPath, len(shortestPath) - 1)

                        # Convert all elements of destinationtypes to tuples if they are lists
                        hashable_etb = tuple(sorted(etb.items())) if isinstance(etb, dict) else etb
                        placementTreeDict[(destinationtypes_hashed, hashable_etb)] = [mySource, MydestinationNodes, shortestPath]
                        routingInfo.append(shortestPath)

                        # update events sent over network
                        for routingNode in shortestPath:
                            if routingNode not in getNodes(etb, eventNodes, IndexEventNodes):
                                setEventNodes(routingNode, etb, eventNodes, IndexEventNodes)
                        newInstance = True

                if newInstance:      
                    myInstance = Instance(etype, etb, [mySource], {projection: routingInfo}) #! #append routing tree information for instance/etb  
                    newInstances.append(myInstance) #!

    if destinationNodes:
        sink_node = destinationNodes[0]

        # Hop-Costs
        #hops = len(find_shortest_path_or_ancestor(routingAlgo, 0, sink_node)) - 1
        hops = len(find_shortest_path_or_ancestor(routingAlgo, 0, sink_node)) - 1 if len(find_shortest_path_or_ancestor(routingAlgo, 0, sink_node)) > 1 else 0

        costs += max(hops, 0)

    return costs, longestPath, newInstances





def NEWcomputeMSplacementCosts_Path(self, projection, sourcetypes, destinationtypes, noFilter, G): #for PathVariant - fix generate EvalPlan
    costs = 0
    destinationNodes = []     
    IndexEventNodes = self.h_IndexEventNodes
    projrates = self.h_projrates
    projFilterDict = self.h_projFilterDict
    mycombi = self.h_mycombi
    singleSelectivities = self.single_selectivity
    rates = self.h_rates_data

    for etype in destinationtypes:
        for etb in IndexEventNodes[etype]:
            destinationNodes += getNodes(etb)            
            
    newInstances = [] #!
    longestPath = 0        
    etype = sourcetypes[0]
    routingInfo = []
        
    for etb in IndexEventNodes[etype]: #parallelize
        newInstance = False   
        MydestinationNodes = list(set(destinationNodes).difference(set(getNodes(etb)))) #only consider nodes that do not already hold etb
        if MydestinationNodes:     
                for dest in MydestinationNodes:
                   if not dest in getNodes(etb):  
                        #are there ms nodes which did not receive etb before
                        node = findBestSource(self, getNodes(etb), [dest]) #best source is node closest to a node of destinationNodes
                        
                        shortestPath = nx.shortest_path(G, dest, node, method='dijkstra') 
                        if len(shortestPath) > longestPath:
                            longestPath = len(shortestPath)
                    
                        if etype in projFilterDict.keys() and  getMaximalFilter(projFilterDict, etype, noFilter): #case input projection has filter
                            mycosts =  len(shortestPath) * getDecomposedTotal(getMaximalFilter(projFilterDict, etype, noFilter), etype)                    
                            if len(IndexEventNodes[etype]) > 1 : # filtered projection has ms placement
                                 partType = returnPartitioning(etype, mycombi[etype])[0]                     
                                 mycosts -= len(shortestPath)  * rates[partType] * singleSelectivities[getKeySingleSelect(partType, etype)] * len(IndexEventNodes[etype])
                                 mycosts += len(shortestPath)  * rates[partType] * singleSelectivities[getKeySingleSelect(partType, etype)] 
                        elif len(etype) == 1:
                            mycosts = len(shortestPath) * rates[etype]
                        else:                    
                            num = NumETBsByKey(etb, etype)                 
                            mycosts = len(shortestPath) *  projrates[etype][1] * num          
                        costs +=  mycosts     

                        routingInfo.append(shortestPath)    # destinations have different sources
                    
                        for routingNode in shortestPath:
                            if not routingNode in getNodes(etb):
                                setEventNodes(routingNode, etb)  
                        newInstance = True        
        if newInstance:
            myInstance = Instance(etype, etb, [node], {projection: routingInfo}) #! #append routing tree information for instance/etb                    
            newInstances.append(myInstance) #!        
    return costs, longestPath, newInstances

# def NEWcomputeMSplacementCosts(projection, sourcetypes, destinationtypes, noFilter): #we need tuples, (C, [E,A]) C should be sent to all e and a nodes ([D,E], [A]) d and e should be sent to all a nodes etc
#     #print(projection, sourcetypes)

    
#     costs = 0
#     destinationNodes = []     

#     for etype in destinationtypes:
#         for etb in IndexEventNodes[etype]:
#             destinationNodes += getNodes(etb)
            
            
#     newInstances = [] #!
#     pathLength = 0        
#     etype = sourcetypes[0]
    
#     f = open("msFilter.txt", "w")
#     if etype in projFilterDict.keys() and getMaximalFilter(projFilterDict, etype, noFilter):
#         print("hasFilter")
#         f.write("VAR=true")
#     else:        
#         f.write("VAR=false")
#     f.close()     
        
        
#     for etb in IndexEventNodes[etype]: #parallelize 
            
       
#             MydestinationNodes = list(set(destinationNodes).difference(set(getNodes(etb)))) #only consider nodes that do not already hold etb
#             if MydestinationNodes: #are there ms nodes which did not receive etb before
#                     node = findBestSource(getNodes(etb), MydestinationNodes) #best source is node closest to a node of destinationNodes
#                     treenodes = copy.deepcopy(MydestinationNodes) 
#                     treenodes.append(node)
#                     from networkx.algorithms.approximation import steiner_tree
#                     mytree = steiner_tree(G, treenodes)
                    
#                     myInstance = Instance(etype, etb, [node], {projection: list(mytree.edges)}) #! #append routing tree information for instance/etb                    
#                     newInstances.append(myInstance) #!
                    
                    
#                     myPathLength = max([len(nx.shortest_path(mytree, x, node, method='dijkstra')) for x in MydestinationNodes]) - 1
                    
                    
#                     if etype in projFilterDict.keys() and  getMaximalFilter(projFilterDict, etype, noFilter): #case input projection has filter
#                         mycosts =  len(mytree.edges()) * getDecomposedTotal(getMaximalFilter(projFilterDict, etype, noFilter), etype)                    
#                         if len(IndexEventNodes[etype]) > 1 : # filtered projection has ms placement
#                              partType = returnPartitioning(etype, mycombi[etype])[0]                     
#                              mycosts -= len(mytree.edges())  * rates[partType] * singleSelectivities[getKeySingleSelect(partType, etype)] * len(IndexEventNodes[etype])
#                              mycosts += len(mytree.edges())  * rates[partType] * singleSelectivities[getKeySingleSelect(partType, etype)] 
#                     elif len(etype) == 1:
#                         mycosts = len(mytree.edges()) * rates[etype]
#                     else:                    
#                         num = NumETBsByKey(etb, etype)                 
#                         mycosts = len(mytree.edges()) *  projrates[etype][1] * num     # FILTER              
                        
#                     placementTreeDict[(tuple(destinationtypes),etb)] = [node, MydestinationNodes, mytree] #only kept for updating in the next step
#                     costs +=  mycosts 
                    
#                     pathLength = max([pathLength, myPathLength])
                    
#                     # update events sent over network
#                     for routingNode in mytree.nodes():
#                         if not routingNode in getNodes(etb):
#                             setEventNodes(routingNode, etb)
    
        
#   # print(list(map(lambda x: str(x), sourcetypes)), costs)           
#     return costs, pathLength, newInstances


def findBestSource(self, sources, actualDestNodes): #this is only a heuristic, as the closest node can still be shit with respect to a good steiner tree ?+
    allPairs = self.allPairs
    curmin = np.inf     
    for node in sources:        
        if min([allPairs[node][x] for x in actualDestNodes]) < curmin:
           curmin =  min([allPairs[node][x] for x in actualDestNodes])
           bestSource = node
    return bestSource

# def findBestSource(self, sources, actualDestNodes, allowedSources=None):
#     """
#     Wählt die beste Quelle aus einer Liste `sources`, die am nächsten zu einem Ziel in `actualDestNodes` liegt.
#     Optional: Nur Quellen aus `allowedSources` werden berücksichtigt.
#     """
#     allPairs = self.allPairs
#     curmin = np.inf
#     bestSource = None

#     # Falls eine Filtermenge angegeben ist, filtere die Quellen
#     filtered_sources = [s for s in sources if allowedSources is None or s in allowedSources]

#     for node in filtered_sources:
#         min_dist = min([allPairs[node][x] for x in actualDestNodes])
#         if min_dist < curmin:
#             curmin = min_dist
#             bestSource = node

#     return bestSource

        
# def getDestinationsUpstream(projection):
#     return  range(len(allPairs))       
        
def ComputeSingleSinkPlacement(projection, combination, noFilter,projFilterDict,EventNodes,IndexEventNodes,network_data,allPairs,mycombi,rates,singleSelectivities,projrates,Graph,network):
    from allPairs import create_routing_dict
    routingDict = create_routing_dict(Graph)
    costs = np.inf
    node = 0
    Filters  = []    
   # routingDict = dict(nx.all_pairs_shortest_path(Graph))
    #print(routingDict)
    routingAlgo = dict(nx.all_pairs_shortest_path(Graph))
     
    # add filters of projections to eventtpes in combi, if filters added, use costs of filter -> compute costs for single etbs of projrates 
    intercombi = []
    ##### FILTERS
    for proj in combination:
        intercombi.append(proj)  
        #print(list(map(lambda x: str(x), list(projFilterDict.keys()))))
        if len(proj) > 1 and len(getMaximalFilter(projFilterDict, proj, noFilter)) > 0:            
            Filters.append((proj,getMaximalFilter(projFilterDict, proj, noFilter)))            
            for etype in getMaximalFilter(projFilterDict, proj, noFilter):
                intercombi.append(etype)
    combination = list(set(intercombi))
    
    myProjection = Projection(projection, {}, [], [], Filters) #!
    print("Network")
    print(network)
    print(network_data)
    # Extract only the keys (nodes) with an empty list of connections
    # iterate through less nodes if possible
    non_leaf = [node for node, neighbors in network_data.items() if not neighbors and network[node].computational_power >= projection.computing_requirements]
    
    for destination in non_leaf:
        # consider relevant nodes for placement
        # if not Graph.nodes[destination].get('relevant', False):  # Out of calculation if not relevant
        #     continue
        skip_destination = False  # Flag to determine if we should skip this destination
        for eventtype in combination:
            for etb in IndexEventNodes[eventtype]:
                possibleSources = getNodes(etb,EventNodes,IndexEventNodes)
                for source in possibleSources:
                    # Use the routing_dict to get the common ancestor
                    common_ancestor = routingDict[destination][source]['common_ancestor']
                    if common_ancestor != destination:
                        # print(f"Skipping destination {destination} for source {source} (Common ancestor: {common_ancestor})")
                        skip_destination = True
                        break
                if skip_destination:
                    break  # Break out of the etb loop
            if skip_destination:
                break  # Break out of the eventtype loop
        if skip_destination:
            continue  # Move on to the next destination without computing costs
        
        mycosts = 0
        for eventtype in combination:
                # if not Graph.nodes[destination].get('relevant', False):  # Out of calculation if not relevant
                #     continue
                for etb in IndexEventNodes[eventtype]: #check for all sources #here iterated over length of IndesEventNodes to get all sources for etb Instances
                        
                        possibleSources = getNodes(etb,EventNodes,IndexEventNodes)
                        mySource = possibleSources[0]
                        for source in possibleSources:
                            if allPairs[destination][source] < allPairs[destination][mySource]:                               
                                   mySource  = source
                        "check for computing power"
                        #print(hops)
                        if eventtype in projFilterDict.keys() and  getMaximalFilter(projFilterDict, eventtype, noFilter): #case filter 
                            mycosts +=  allPairs[destination][mySource] * getDecomposedTotal(getMaximalFilter(projFilterDict, eventtype, noFilter), eventtype)                    
                            if len(IndexEventNodes[eventtype]) > 1 : # filtered projection has ms placement
                                partType = returnPartitioning(eventtype, mycombi[eventtype])[0]                     
                                mycosts -= allPairs[destination][mySource] * rates[partType] * singleSelectivities[getKeySingleSelect(partType, eventtype)] * len(IndexEventNodes[eventtype])
                                mycosts += allPairs[destination][mySource] * rates[partType] * singleSelectivities[getKeySingleSelect(partType, eventtype)]
                        elif eventtype in rates.keys():        # case primitive event

                            mycosts += (rates[eventtype] * allPairs[destination][mySource] )
                        else: # case projection                         
                             num = NumETBsByKey(etb, eventtype,IndexEventNodes)
                             mycosts += projrates[eventtype][1] * allPairs[destination][mySource] * num 
        print(mycosts)
        if mycosts < costs:
            costs = mycosts
            node = destination
    myProjection.addSinks(node) #!

    # Remove computational power of sink for next iteration
    # if network[node].computational_power >= projection.computing_requirements:
    #     network[node].computational_power -= projection.computing_requirements

    newInstances = [] #!
    # Update Event Node Matrice, by adding events etbs sent to node through node x to events of node x
    longestPath  = 0
    for eventtype in combination:
            curInstances = [] #!
            for etb in IndexEventNodes[eventtype]:
                possibleSources = getNodes(etb,EventNodes,IndexEventNodes)
                mySource = possibleSources[0] #??
                for source in possibleSources:                    
                    if allPairs[node][source] < allPairs[node][mySource]:
                       mySource  = source     
                
                shortestPath = find_shortest_path_or_ancestor(routingAlgo, mySource, node) 
              
                if len(shortestPath) - 1 > longestPath:
                    longestPath = len(shortestPath) - 1                    
                newInstance = Instance(eventtype, etb, [mySource], {projection: shortestPath}) #!
                curInstances.append(newInstance) #!    
                
                for stop in shortestPath:
                    if not stop in getNodes(etb,EventNodes,IndexEventNodes):                        
                        setEventNodes(stop, etb,EventNodes,IndexEventNodes) 
                        
            newInstances += curInstances          #!   
            myProjection.addInstances(eventtype, curInstances)     #!        
                        
    SiSManageETBs(projection, node,IndexEventNodes,EventNodes,network_data)
    hops = len(find_shortest_path_or_ancestor(routingAlgo, 0, node)) - 1 if len(find_shortest_path_or_ancestor(routingAlgo, 0, node)) > 1 else 0
    myProjection.addSpawned([IndexEventNodes[projection][0]]) #!
    costs += hops
    return costs, node, longestPath, myProjection, newInstances, Filters

# def costsAt(eventtype, node):
#     mycosts = 0
#     for etb in IndexEventNodes[eventtype]:
#                 possibleSources = getNodes(etb)
#                 mySource = possibleSources[0]
#                 for source in possibleSources:
#                     if allPairs[node][source] <= allPairs[node][mySource]:
#                        mySource  = source
#                 mycosts += rates[eventtype] * allPairs[node][mySource] 
#     return mycosts

def NEWcomputeCentralCosts(workload,IndexEventNodes,allPairs,rates,EventNodes,G):
    #Adding all Eventtypes (simple events) to the list
    import networkx as nx
    eventtypes = []
    for i in workload:
        myevents = i.leafs()
        #print(myevents)
        for e in myevents:
            # print(e)
            # print("Eventtypes")
            # print(eventtypes)
            if not e in eventtypes:
                eventtypes.append(e)
    #print(eventtypes)
    costs = np.inf
    node = 0
    # for destination in range(len(allPairs)):
    destination = 0
    mycosts = 0      
    for eventtype in eventtypes:
        oldcosts = mycosts
        for etb in IndexEventNodes[eventtype]:

            possibleSources = getNodes(etb,EventNodes,IndexEventNodes)
            mySource = possibleSources[0]
            for source in possibleSources:
                
                if allPairs[destination][source] <= allPairs[destination][mySource]:
                    mySource  = source
            mycosts += rates[eventtype] * allPairs[destination][mySource]       
    if mycosts < costs:
        costs = mycosts
        node = destination
    longestPath = max(allPairs[node])
    
    routingDict = {} # for evaluation plan
    for e in eventtypes:        
        routingDict[e] = {}
        for etb in IndexEventNodes[e]:
            possibleSources = getNodes(etb,EventNodes,IndexEventNodes)
            mySource = possibleSources[0]
            shortestPath = nx.shortest_path(G, mySource, node, method='dijkstra')  
            routingDict[e][etb] = shortestPath
 
    for eventtype in eventtypes:    
        thiscosts = 0
        for etb in IndexEventNodes[eventtype]:
                    mySource = getNodes(etb,EventNodes,IndexEventNodes)[0]         
                    thiscosts += rates[eventtype] * allPairs[node][mySource]
                    #print(allPairs[node][mySource])
        #print(eventtype, thiscosts )    
    return (costs, node, longestPath, routingDict) 
                    
#TODO compute and print rates saved by placement