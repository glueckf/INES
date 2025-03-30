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



# def findBestSource(sources, actualDestNodes): #this is only a heuristic, as the closest node can still be shit with respect to a good steiner tree ?+
#     curmin = np.inf     
#     for node in sources:        
#         if min([allPairs[node][x] for x in actualDestNodes]) < curmin:
#            curmin =  min([allPairs[node][x] for x in actualDestNodes])
#            bestSource = node
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
    non_leaf = [node for node, neighbors in network_data.items() if not neighbors and network[node].computational_power >= projection.computing_requirements]
    
    for destination in non_leaf:
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
                  #  print(allPairs[node][mySource])
      #  print(eventtype, thiscosts )    
    return (costs, node, longestPath, routingDict) 
                    
#TODO compute and print rates saved by placement