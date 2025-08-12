#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:16:11 2021

@author: samira
"""
import math
import multiprocessing

import networkx
#from helper.processCombination_aug import *
from helper.filter import getMaximalFilter, getDecomposedTotal
from helper.structures import getNodes, NumETBsByKey, setEventNodes, SiSManageETBs
from projections import returnPartitioning
from functools import partial
from allPairs import find_shortest_path_or_ancestor
import copy
from EvaluationPlan import Instance, Projection
import numpy as np
from helper.filter import getKeySingleSelect
import networkx as nx
from helper.structures import MSManageETBs, getETBs
from networkx.algorithms.approximation import steiner_tree
from prepp import generate_prePP

from src.EvaluationPlan import EvaluationPlan


def getFilters(self, projection, partType):  # move to filter file eventually
    IndexEventNodes = self.h_IndexEventNodes
    projrates = self.h_projrates
    eventNodes = self.h_eventNodes
    totalETBs = 0
    for etb in IndexEventNodes[partType]:  #for each multi-sink

        numETBs = 1
        node = getNodes(etb, eventNodes, IndexEventNodes)[0]
        myETBs = getETBs(node, eventNodes, IndexEventNodes)
        if not set(IndexEventNodes[projection]).issubset(set(getETBs(node, eventNodes,
                                                                     IndexEventNodes))):  # it is  checked if the node already received all etbs of the projection, if this is the case its not necessary to reduce something here
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
    Filters = []

    ##### FILTERS, append maximal filters
    intercombi = []

    automaticFilters = 0
    for proj in combination:
        if len(proj) > 1 and len(IndexEventNodes[
                                     proj]) == 1:  #here a node can only have already ALL events if the node is a sink for the projection, this case is however already covered in normal placement cost calculation
            automaticFilters += getFilters(proj, partType[0])  # TODO first version

        intercombi.append(proj)
        if len(proj) > 1 and len(getMaximalFilter(projFilterDict, proj,
                                                  noFilter)) > 0:  # those are the extra events that need to be sent around due to filters
            Filters.append((proj, getMaximalFilter(projFilterDict, proj, noFilter)))
            #print("Using Filter: " + str(getMaximalFilter(projFilterDict, proj)) + ": " + str(projFilterDict[proj][getMaximalFilter(projFilterDict, proj)][0]) + " instead of " + str(projrates[proj])  )
            for etype in getMaximalFilter(projFilterDict, proj, noFilter):
                intercombi.append(etype)
    combination = list(set(intercombi))
    myPathLength = 0

    totalInstances = []  #!
    myNodes = []

    # for eventtype in mycombi.get(projection, []):
    #     if eventtype not in IndexEventNodes:
    #         continue
    #     for etb in IndexEventNodes[eventtype]:
    #         candidates = getNodes(etb, eventNodes, IndexEventNodes)
    #         for node in candidates:
    #             if self.network[node].computational_power >= projection.computing_requirements:
    #                 myNodes.append(node)

    # # Remove duplicates
    # myNodes = list(set(myNodes))

    # # Fallback if no valid nodes found
    # if not myNodes:
    #     print(f"[Warning] No valid nodes found for projection {projection}, using fallback node 0")
    #     myNodes = [0]
    # sinkNodes = []

    myProjection = Projection(projection, {}, [], [], Filters)  #!
    for myInput in combination:
        if not partType[0] == myInput:
            if myInput in sharedDict.keys():
                #result = NEWcomputeMSplacementCosts_Path(projection, [myInput], sharedDict[myInput], noFilter)
                result = NEWcomputeMSplacementCosts(self, projection, [myInput], sharedDict[myInput], noFilter, G)
                costs += result[0]  #fix SharedDict with Filter Inputs

            else:
                #result = NEWcomputeMSplacementCosts_Path(projection, [myInput],  partType[0], noFilter)
                result = NEWcomputeMSplacementCosts(self, projection, [myInput], partType[0], noFilter, G)

                costs += result[0]
            if result[1] > myPathLength:
                myPathLength = result[1]

            myProjection.addInstances(myInput, result[2])  #!
            totalInstances += result[2]  #!
        else:
            myInstances = [Instance(partType[0], partType[0], nodes[partType[0]], {})]
            myProjection.addInstances(partType[0], myInstances)

    # here generate an instance of etbs per parttype and add one line per instance
    MSManageETBs(self, projection, partType[0])

    spawnedInstances = IndexEventNodes[projection]
    myProjection.addSpawned(spawnedInstances)
    for sink in result[3].sinks:
        myProjection.addSinks(sink)
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
    node = []
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
    dest = 0
    costs_per_sink = {}

    # Check filter
    etype = sourcetypes[0]
    with open("msFilter.txt", "w") as f:
        if etype in projFilterDict and getMaximalFilter(projFilterDict, etype, noFilter):
            f.write("VAR=true")
        else:
            f.write("VAR=false")

    myProjection = Projection(projection, {}, [], [], noFilter)  #!
    # search valid sinks
    destinationtypes = [node for node, neighbors in self.h_network_data.items() if
                        not neighbors and self.network[node].computational_power >= projection.computing_requirements]
    # Ensure hashable type for later usage in placementTreeDict
    destinationtypes_hashed = tuple(destinationtypes)
    # Filters all valid destinationNodes
    destinationNodes = []
    for destination in destinationtypes:
        skip = False  # Flag to determine if we should skip this destination
        for etype in mycombi.get(projection, []):
            # if eventtype not in IndexEventNodes:
            #     continue
            for etb in IndexEventNodes.get(etype, []):
                possibleSources = getNodes(etb, eventNodes, IndexEventNodes)
                for source in possibleSources:
                    # Use the routing_dict to get the common ancesto
                    common_ancestor = routingDict[destination][source]['common_ancestor']
                    if common_ancestor != destination:
                        skip = True
                        break
                if skip:
                    break  # Break out of the etb loop
            if skip:
                break  # Break out of the eventtype loop
        if skip:
            continue
        destinationNodes.append(destination)

        mycosts = 0
        # Main logic: for all event types that belong to the projection
    for etype in mycombi.get(projection, []):
        # if etype not in IndexEventNodes:  # Out of calculation if not relevant
        #     continue
        for etb in IndexEventNodes.get(etype, []):
            newInstance = False
            currentSources = getNodes(etb, eventNodes, IndexEventNodes)
            MydestinationNodes = list(set(destinationNodes) - set(currentSources))
            if MydestinationNodes:
                for dest in MydestinationNodes:
                    if not dest in getNodes(etb, eventNodes, IndexEventNodes):
                        #node.append(destinationNodes)
                        mySource = currentSources[0]
                        for source in currentSources:
                            if allPairs[dest][source] < allPairs[dest][mySource]:
                                mySource = source
                        #node = findBestSource(self,mySource,dest)
                        # shortestPath = find_shortest_path_or_ancestor(routingAlgo, mySource, dest)
                        # if not shortestPath or not isinstance(shortestPath, list):
                        #     print(f"[Warning] No edge from {mySource} to {dest} for etb {etb}")
                        #     continue
                        # edges = list(zip(shortestPath[:-1], shortestPath[1:]))

                        #print(f"[Tracking] send {etb} from {mySource} to {dest} over {shortestPath}")
                        # Cost calculation
                        if etype in projFilterDict.keys() and getMaximalFilter(projFilterDict, etype,
                                                                               noFilter):  #case input projection has filter
                            mycosts = allPairs[dest][mySource] * getDecomposedTotal(
                                getMaximalFilter(projFilterDict, etype, noFilter), type)
                            if len(IndexEventNodes[etype]) > 1:  # filtered projection has ms placement
                                partType = returnPartitioning(etype, mycombi[etype])[0]
                                mycosts -= allPairs[dest][mySource] * rates[partType] * singleSelectivities[
                                    getKeySingleSelect(partType, etype)] * len(IndexEventNodes[etype])
                                mycosts += allPairs[dest][mySource] * rates[partType] * singleSelectivities[
                                    getKeySingleSelect(partType, etype)]
                        elif len(etype) == 1:
                            mycosts = allPairs[dest][mySource] * rates[etype]
                        else:
                            num = NumETBsByKey(etb, etype, IndexEventNodes)
                            mycosts = allPairs[dest][mySource] * projrates[etype][1] * num  # FILTER
                            # pathlength and costs
                            #print(mycosts)
        #costs += mycosts
        if MydestinationNodes:
            costs += mycosts
    #print(mycosts)
    # if mycosts < costs and mycosts != 0:
    #     costs = mycosts
    #print(mycosts)
    # if mycosts != 0:
    #     costs += mycosts
    #costs += mycosts
    #desti = dest
    for de in destinationNodes:
        node.append(de)
    for n in node:
        myProjection.addSinks(n)

    #print(f"[MS] Using sink {node} for projection {projection}")

    for etype in mycombi.get(projection, []):
        curInstances = []  #!
        for etb in IndexEventNodes[etype]:
            # MydestinationNodes = list(set(destinationNodes) - set(currentSources))
            # if MydestinationNodes:
            #         for dest in MydestinationNodes:
            #             if not dest in getNodes(etb, eventNodes, IndexEventNodes):
            #                  continue
            possibleSources = getNodes(etb, eventNodes, IndexEventNodes)
            mySource = possibleSources[0]  #??
            for source in possibleSources:
                if allPairs[destination][source] < allPairs[destination][mySource]:
                    mySource = source
            shortestPath = find_shortest_path_or_ancestor(routingAlgo, mySource, destination)

            if len(shortestPath) - 1 > longestPath:
                longestPath = len(shortestPath) - 1
            newInstance = Instance(etype, etb, [mySource], {projection: shortestPath})  #!
            curInstances.append(newInstance)  #!
            hashable_etb = tuple(sorted(etb.items())) if isinstance(etb, dict) else etb
            placementTreeDict[(destinationtypes_hashed, hashable_etb)] = [mySource, destination, shortestPath]

            for stop in shortestPath:
                if not stop in getNodes(etb, eventNodes, IndexEventNodes):
                    setEventNodes(stop, etb, eventNodes, IndexEventNodes)

        newInstances += curInstances  #!
        myProjection.addInstances(etype, curInstances)  #!                      # newInstance = True
        # if newInstance:
        #     myInstance = Instance(etype, etb, [mySource], {projection: routingInfo}) #! #append routing tree information for instance/etb
        #     newInstances.append(myInstance) #!
    if destinationNodes:
        sink_node = destinationNodes[0]
        #MSManageETBs(self, projection, partType[0]) 
        # Hop-Costs
        #hops = len(find_shortest_path_or_ancestor(routingAlgo, 0, sink_node)) - 1
        hops = len(find_shortest_path_or_ancestor(routingAlgo, 0, sink_node)) - 1 if len(
            find_shortest_path_or_ancestor(routingAlgo, 0, sink_node)) > 1 else 0
        #myProjection.addSpawned([IndexEventNodes[projection][0]]) #!
        costs += max(hops, 0)
    return costs, longestPath, newInstances, myProjection


def NEWcomputeMSplacementCosts_Path(self, projection, sourcetypes, destinationtypes, noFilter,
                                    G):  #for PathVariant - fix generate EvalPlan
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

    newInstances = []  #!
    longestPath = 0
    etype = sourcetypes[0]
    routingInfo = []

    for etb in IndexEventNodes[etype]:  #parallelize
        newInstance = False
        MydestinationNodes = list(
            set(destinationNodes).difference(set(getNodes(etb))))  #only consider nodes that do not already hold etb
        if MydestinationNodes:
            for dest in MydestinationNodes:
                if not dest in getNodes(etb):
                    #are there ms nodes which did not receive etb before
                    node = findBestSource(self, getNodes(etb),
                                          [dest])  #best source is node closest to a node of destinationNodes

                    shortestPath = nx.shortest_path(G, dest, node, method='dijkstra')
                    if len(shortestPath) > longestPath:
                        longestPath = len(shortestPath)

                    if etype in projFilterDict.keys() and getMaximalFilter(projFilterDict, etype,
                                                                           noFilter):  #case input projection has filter
                        mycosts = len(shortestPath) * getDecomposedTotal(
                            getMaximalFilter(projFilterDict, etype, noFilter), etype)
                        if len(IndexEventNodes[etype]) > 1:  # filtered projection has ms placement
                            partType = returnPartitioning(etype, mycombi[etype])[0]
                            mycosts -= len(shortestPath) * rates[partType] * singleSelectivities[
                                getKeySingleSelect(partType, etype)] * len(IndexEventNodes[etype])
                            mycosts += len(shortestPath) * rates[partType] * singleSelectivities[
                                getKeySingleSelect(partType, etype)]
                    elif len(etype) == 1:
                        mycosts = len(shortestPath) * rates[etype]
                    else:
                        num = NumETBsByKey(etb, etype)
                        mycosts = len(shortestPath) * projrates[etype][1] * num
                    costs += mycosts

                    routingInfo.append(shortestPath)  # destinations have different sources

                    for routingNode in shortestPath:
                        if not routingNode in getNodes(etb):
                            setEventNodes(routingNode, etb)
                    newInstance = True
        if newInstance:
            myInstance = Instance(etype, etb, [node], {
                projection: routingInfo})  #! #append routing tree information for instance/etb
            newInstances.append(myInstance)  #!
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


def findBestSource(self, sources,
                   actualDestNodes):  #this is only a heuristic, as the closest node can still be shit with respect to a good steiner tree ?+
    allPairs = self.allPairs
    curmin = np.inf
    for node in sources:
        if min([allPairs[node][x] for x in actualDestNodes]) < curmin:
            curmin = min([allPairs[node][x] for x in actualDestNodes])
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

def ComputeSingleSinkPlacement(projection, combination, noFilter, projFilterDict, EventNodes, IndexEventNodes,
                               network_data, allPairs, mycombi, rates, singleSelectivities, projrates, Graph, network):
    from allPairs import create_routing_dict
    routingDict = create_routing_dict(Graph)
    costs = np.inf
    node = 0
    Filters = []
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
            Filters.append((proj, getMaximalFilter(projFilterDict, proj, noFilter)))
            for etype in getMaximalFilter(projFilterDict, proj, noFilter):
                intercombi.append(etype)
    combination = list(set(intercombi))

    myProjection = Projection(projection, {}, [], [], Filters)  #!
    # print(f"[NETWORK DEBUG] Network nodes: {len(network)} total")
    # print(f"[NETWORK DEBUG] Network data structure: {list(network_data.keys())[:5]}..." if len(
    #     network_data) > 5 else f"[NETWORK DEBUG] Network data: {network_data}")
    # print(
    #     f"[NETWORK DEBUG] Sample network info: {[(i, getattr(node, 'computational_power', 'N/A')) for i, node in enumerate(network[:3])]}" if isinstance(
    #         network, list) else f"[NETWORK DEBUG] Network type: {type(network)}")
    # Extract only the keys (nodes) with an empty list of connections
    # iterate through less nodes if possible
    non_leaf = [node for node, neighbors in network_data.items() if
                not neighbors and network[node].computational_power >= projection.computing_requirements]

    for destination in non_leaf:
        # consider relevant nodes for placement
        # if not Graph.nodes[destination].get('relevant', False):  # Out of calculation if not relevant
        #     continue
        skip_destination = False  # Flag to determine if we should skip this destination
        for eventtype in combination:
            for etb in IndexEventNodes[eventtype]:
                possibleSources = getNodes(etb, EventNodes, IndexEventNodes)
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
            for etb in IndexEventNodes[
                eventtype]:  #check for all sources #here iterated over length of IndesEventNodes to get all sources for etb Instances

                possibleSources = getNodes(etb, EventNodes, IndexEventNodes)
                mySource = possibleSources[0]
                for source in possibleSources:
                    if allPairs[destination][source] < allPairs[destination][mySource]:
                        mySource = source
                "check for computing power"
                #print(hops)
                if eventtype in projFilterDict.keys() and getMaximalFilter(projFilterDict, eventtype,
                                                                           noFilter):  #case filter
                    mycosts += allPairs[destination][mySource] * getDecomposedTotal(
                        getMaximalFilter(projFilterDict, eventtype, noFilter), eventtype)
                    if len(IndexEventNodes[eventtype]) > 1:  # filtered projection has ms placement
                        partType = returnPartitioning(eventtype, mycombi[eventtype])[0]
                        mycosts -= allPairs[destination][mySource] * rates[partType] * singleSelectivities[
                            getKeySingleSelect(partType, eventtype)] * len(IndexEventNodes[eventtype])
                        mycosts += allPairs[destination][mySource] * rates[partType] * singleSelectivities[
                            getKeySingleSelect(partType, eventtype)]
                elif eventtype in rates.keys():  # case primitive event

                    mycosts += (rates[eventtype] * allPairs[destination][mySource])
                else:  # case projection
                    num = NumETBsByKey(etb, eventtype, IndexEventNodes)
                    mycosts += projrates[eventtype][1] * allPairs[destination][mySource] * num
        # print(f"[COSTS] Current mycosts: {mycosts}")
        if mycosts < costs:
            costs = mycosts
            node = destination
    myProjection.addSinks(node)  #!

    # Remove computational power of sink for next iteration
    # if network[node].computational_power >= projection.computing_requirements:
    #     network[node].computational_power -= projection.computing_requirements

    newInstances = []  #!
    # Update Event Node Matrice, by adding events etbs sent to node through node x to events of node x
    longestPath = 0
    for eventtype in combination:
        curInstances = []  #!
        for etb in IndexEventNodes[eventtype]:
            possibleSources = getNodes(etb, EventNodes, IndexEventNodes)
            mySource = possibleSources[0]  #??
            for source in possibleSources:
                if allPairs[node][source] < allPairs[node][mySource]:
                    mySource = source

            shortestPath = find_shortest_path_or_ancestor(routingAlgo, mySource, node)

            if len(shortestPath) - 1 > longestPath:
                longestPath = len(shortestPath) - 1
            newInstance = Instance(eventtype, etb, [mySource], {projection: shortestPath})  #!
            curInstances.append(newInstance)  #!

            for stop in shortestPath:
                if not stop in getNodes(etb, EventNodes, IndexEventNodes):
                    setEventNodes(stop, etb, EventNodes, IndexEventNodes)

        newInstances += curInstances  #!
        myProjection.addInstances(eventtype, curInstances)  #!

    SiSManageETBs(projection, node, IndexEventNodes, EventNodes, network_data)
    hops = len(find_shortest_path_or_ancestor(routingAlgo, 0, node)) - 1 if len(
        find_shortest_path_or_ancestor(routingAlgo, 0, node)) > 1 else 0
    myProjection.addSpawned([IndexEventNodes[projection][0]])  #!
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

def NEWcomputeCentralCosts(workload, IndexEventNodes, allPairs, rates, EventNodes, G):
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

            possibleSources = getNodes(etb, EventNodes, IndexEventNodes)
            mySource = possibleSources[0]
            for source in possibleSources:

                if allPairs[destination][source] <= allPairs[destination][mySource]:
                    mySource = source
            mycosts += rates[eventtype] * allPairs[destination][mySource]
    if mycosts < costs:
        costs = mycosts
        node = destination
    longestPath = max(allPairs[node])

    routingDict = {}  # for evaluation plan
    for e in eventtypes:
        routingDict[e] = {}
        for etb in IndexEventNodes[e]:
            possibleSources = getNodes(etb, EventNodes, IndexEventNodes)
            mySource = possibleSources[0]
            shortestPath = nx.shortest_path(G, mySource, node, method='dijkstra')
            routingDict[e][etb] = shortestPath

    for eventtype in eventtypes:
        thiscosts = 0
        for etb in IndexEventNodes[eventtype]:
            mySource = getNodes(etb, EventNodes, IndexEventNodes)[0]
            thiscosts += rates[eventtype] * allPairs[node][mySource]
            #print(allPairs[node][mySource])
        #print(eventtype, thiscosts )    
    return (costs, node, longestPath, routingDict)


#TODO compute and print rates saved by placement

def compute_operator_placement_with_prepp(
        self,
        projection: dict,
        combination: list,
        no_filter: int,
        proj_filter_dict: dict,
        event_nodes: list,
        index_event_nodes: dict,
        network_data: dict,
        all_pairs: list,
        mycombi: dict,
        rates: dict,
        single_selectivity: dict,
        projrates: dict,
        graph: networkx.Graph,
        network: list,
        central_eval_plan):
    """
    Integrated function to compute operator placement with prepp in an integrated manner.

    Args:
        self: Instance of the class containing all necessary data.
        projection: The projection for which the placement is computed.
        combination: The combination of event types to consider for the placement.
        no_filter: Flag to indicate whether to apply filters or not.
        proj_filter_dict: Dictionary containing filters for projections and ?.
        event_nodes: Matrix mapping event types to nodes.
        IndexEventnodes: Indexed dictionary mapping event types to their respective ETBs.
        network_data: Dictionary containing data on which node produces which event types.
        all_pairs: Matrix containing all pairwise distances between nodes.
        mycombi: Dictionary mapping event types to their combinations.
        rates: Dictionary containing rates for each event type.
        single_selectivity: Dictionary containing selectivity for single event types.
        projrates: Dictionary containing rates for each projection.
        graph: NetworkX graph representing the network topology.
        network: Network object containing all nodes and their respective properties as a dictionary.

    Returns:
        None: As of now, this function does not return any value.

    Comments:
        This function is the first iteration of the integrated operator placement with prepp.

        It filters the available nodes based on the resources available in the network and then calculates the costs for
        each node base on the push pull communication model being chosen.

        The goal is to extract a subgraph from the network for each possible placement of the projection and
        then compute the prepp costs on it.

        Future Iterations could include:
        1. Adding more complex ressource filtering, as projections get computationally lighter when using prepp.
        So nodes with initially not enough resources could become feasible for placement again.
        2. Adding heuristics to sort the projections based on their costs or something else.
        3. Adding different pruning strategies to reduce the search space for the placement.
        4. Handling certain edge cases, such as when one projection is placed on a node with an event needed to be
        pulled and the second projection wanting it to be pushed.
        TODO: Handle mentioned future iterations.

    """

    # Initialize placement state
    placement_state = _initialize_placement_state(
        combination, proj_filter_dict, no_filter, projection, graph
    )
    print(f"[PLACEMENT] Starting placement computation for: {projection}")

    possible_placement_nodes = check_possible_placement_nodes_for_input(
        projection,
        placement_state['extended_combination'],
        network_data,
        network,
        index_event_nodes,
        event_nodes,
        placement_state['routing_dict']
    )

    # Reverse the order of the possible placement nodes to prioritize nodes closer to the source
    possible_placement_nodes.reverse()

    for node in possible_placement_nodes:

        has_enough_resources = check_resources(node, projection, network, combination)

        print(f"[PLACEMENT] Evaluating node {node} with enough resources: {has_enough_resources}")

        subgraph = extract_subgraph(
            node,
            network,
            graph,
            placement_state['extended_combination'],
            index_event_nodes,
            event_nodes,
            placement_state['routing_dict']
        )

        print(f"[PLACEMENT] Extracted subgraph for node {node}: {subgraph}")

        if has_enough_resources:

            costs = calculate_prepp_costs_on_subgraph(self, node, subgraph, projection, central_eval_plan)

        else:
            costs = calculate_all_push_costs_on_subgraph()

        # Update costs if this node is better
        if costs < placement_state['costs']:
            placement_state['costs'] = costs
            placement_state['best_node'] = node

    # Return the best node and its associated costs
    return None


def _initialize_placement_state(combination, proj_filter_dict, no_filter, projection, graph):
    """
    Prepare and return the initial state needed for computing projection placement in the network.

    This function is responsible for:
      1. **Routing information setup** – Precomputes shortest paths and a routing dictionary
         so later placement logic can quickly check relationships between nodes.
      2. **Filter application** – Determines and attaches maximal filters to certain
         event-type combinations if they are relevant, unless `no_filter` is set.
      3. **Extended combination generation** – Expands the given `combination` with
         any additional event types required by those filters.
      4. **Projection object creation** – Builds a `Projection` object holding the
         projection’s identity and its associated filters.

    Args:
        combination (list):
            List of event types (or tuples of event types) that the projection depends on.
        proj_filter_dict (dict):
            Mapping of event-type combinations to their available filters.
        no_filter (bool):
            If True, ignore all filters; if False, apply maximal filters when available.
        projection (str or object):
            Identifier or definition of the projection we are placing.
        graph (networkx.Graph):
            The network topology, where nodes represent processing locations and edges
            represent possible routing paths.

    Returns:
        dict: A dictionary containing the initialized placement state:
            - **routing_dict**: Nested dict describing routing info, including common ancestors.
            - **routing_algo**: Dictionary of all shortest paths between nodes.
            - **filters**: List of `(combination, maximal_filter)` tuples applied.
            - **extended_combination**: The combination plus any added filters (duplicates removed).
            - **projection**: The constructed `Projection` object.
            - **costs**: Initial cost set to `math.inf` (placeholder for later optimization).
            - **best_node**: Initial best node set to `0` (no placement chosen yet).

    Comments:
        N/A
    """
    from allPairs import create_routing_dict
    import networkx as nx

    # Create routing structures
    routing_dict = create_routing_dict(graph)
    routing_algo = dict(nx.all_pairs_shortest_path(graph))

    # Process filters and extend combination
    filters = []
    extended_combination = []

    for proj in combination:
        extended_combination.append(proj)
        if len(proj) > 1 and len(getMaximalFilter(proj_filter_dict, proj, no_filter)) > 0:
            max_filter = getMaximalFilter(proj_filter_dict, proj, no_filter)
            filters.append((proj, max_filter))
            extended_combination.extend(max_filter)

    # Remove duplicates from extended combination
    extended_combination = list(set(extended_combination))

    # Create projection object
    my_projection = Projection(projection, {}, [], [], filters)

    return {
        'routing_dict': routing_dict,
        'routing_algo': routing_algo,
        'filters': filters,
        'extended_combination': extended_combination,
        'projection': my_projection,
        'costs': math.inf,
        'best_node': 0
    }


def check_possible_placement_nodes_for_input(projection, combination, network_data, network, index_event_nodes,
                                             event_nodes, routing_dict):
    """
    Check which non-leaf nodes are suitable for placing a projection based on common ancestor requirements.
    
    This function identifies fog/cloud nodes (non-leaf nodes that don't produce events) where the projection 
    could be placed by checking if the node can be a common ancestor for all required ETB sources.
    No resource checks are performed - only topological placement feasibility is considered.

    # TODO: Here we could also consider to do some prefiltering or add some heuristic to reduce the search space.
    
    Args:
        projection: The projection object for which placement is being computed
        combination: The combination of event types to consider for the placement
        network_data: Dictionary mapping nodes to the event types they produce
        index_event_nodes: Indexed dictionary mapping event types to their respective ETBs
        event_nodes: Matrix mapping event types to nodes
        routing_dict: Dictionary containing routing information with common ancestors
        
    Returns:
        list: List of non-leaf node indices where the projection can be placed
    """
    from helper.structures import getNodes

    print(f"[PLACEMENT] Analyzing placement options for: {projection}")
    print(f"[PLACEMENT] Combination context: {combination}")

    # Get non-leaf nodes (nodes that don't produce events - fog/cloud nodes)
    non_leaf = [node for node, neighbors in network_data.items() if not neighbors]
    print(
        f"[PLACEMENT] Available fog/cloud nodes: {len(non_leaf)} nodes - {non_leaf[:5]}{'...' if len(non_leaf) > 5 else ''}")

    suitable_nodes = []

    # Check each non-leaf node for placement feasibility
    for destination in non_leaf:
        print(f"[PLACEMENT] Evaluating node {destination}...")

        skip_destination = False  # Flag to determine if we should skip this destination

        # Check all event types in the combination
        for eventtype in combination:
            if skip_destination:
                break

            print(f"[PLACEMENT]   Processing event type: {eventtype}")

            # Check all ETBs for this event type
            for etb in index_event_nodes[eventtype]:
                if skip_destination:
                    break

                print(f"[PLACEMENT]     Analyzing ETB: {etb}")
                possible_sources = getNodes(etb, event_nodes, index_event_nodes)
                print(
                    f"[PLACEMENT]     Available sources: {len(possible_sources)} - {possible_sources[:3]}{'...' if len(possible_sources) > 3 else ''}")

                # Check each source for this ETB
                for source in possible_sources:
                    # Use the routing_dict to get the common ancestor
                    common_ancestor = routing_dict[destination][source]['common_ancestor']
                    print(
                        f"[Placement]       Source {source} -> Destination {destination}, Common ancestor: {common_ancestor}")

                    if common_ancestor != destination:
                        print(
                            f"[Placement]       SKIP: Node {destination} cannot be common ancestor for source {source}")
                        skip_destination = True
                        break

                if skip_destination:
                    break  # Break out of the etb loop

            if skip_destination:
                break  # Break out of the eventtype loop

        if not skip_destination:
            suitable_nodes.append(destination)
            print(f"[PLACEMENT] Node {destination}: SUITABLE - Can serve as common ancestor")
        else:
            print(f"[PLACEMENT] Node {destination}: SKIPPED - Cannot be common ancestor")

    print(f"[PLACEMENT] Result: {len(suitable_nodes)} suitable nodes found: {suitable_nodes}")
    return suitable_nodes


def check_resources(node: int, projection: dict, network: list, combination: list) -> bool:
    """
    Function to check if a node has enough resources to place a projection.
    Checks computational power for all-push scenario and memory for push-pull scenario.
    
    Args:
        node: Node ID to check resources for
        projection: Projection object with computing_requirements
        network: List of network nodes with computational_power and memory attributes
        combination: List of projections in the combination (unused for now)
    
    Returns:
        bool: True if node has sufficient resources, False otherwise

    Comments:
        As of now, this function checks the given computing_requirements for a projection.
        In the future, they should probably be calculated based on the actual inputs of the projection.
        TODO: Add more complex computing requirements calculation.

    """

    # Get the node from network
    if node >= len(network) or node < 0:
        return False

    target_node = network[node]

    try:
        # Check if projection has computing requirements attribute
        if not hasattr(projection, 'computing_requirements'):
            return False

        # Check computational power for all-push scenario
        if target_node.computational_power < projection.computing_requirements:
            return False

        # Check memory for push-pull scenario (needs 2x projection requirements)
        if target_node.memory < (2 * projection.computing_requirements):
            return False

        return True
    except Exception as e:
        print(f"[ERROR] Node {node} does not have required attributes: {e}")
        return False


def extract_subgraph(placement_node, network, graph, combination, index_event_nodes, event_nodes, routing_dict):
    """
    Extract a subgraph for a given placement node containing all relevant nodes and paths
    needed for routing event data to that node.
    
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

    Comments:
        As of now, this function implements a basic extraction of relevant nodes and paths like in original.
        In the future this could be improved by adding simpler filter techniques and existing structures.
        TODO: Improve extraction logic.
    """
    import networkx as nx
    from helper.structures import getNodes

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
                        try:
                            # Get shortest path from routing dict or compute it
                            if placement_node in routing_dict and source_node in routing_dict[placement_node]:
                                path_info = routing_dict[placement_node][source_node]
                                if 'path' in path_info:
                                    path_nodes = path_info['path']
                                elif 'shortest_path' in path_info:
                                    path_nodes = path_info['shortest_path']
                                else:
                                    # Fallback: compute shortest path
                                    path_nodes = nx.shortest_path(graph, source_node, placement_node)
                            else:
                                # Fallback: compute shortest path
                                path_nodes = nx.shortest_path(graph, source_node, placement_node)

                            relevant_nodes.update(path_nodes)
                        except (nx.NetworkXNoPath, KeyError):
                            # If no path exists, just add the source node
                            relevant_nodes.add(source_node)

    # Create subgraph with relevant nodes
    subgraph = graph.subgraph(relevant_nodes).copy()

    # Create node mappings (original ID -> subgraph ID)
    relevant_nodes_list = sorted(list(relevant_nodes))
    node_mapping = {orig_id: new_id for new_id, orig_id in enumerate(relevant_nodes_list)}
    reverse_mapping = {new_id: orig_id for orig_id, new_id in node_mapping.items()}

    # Create remapped subgraph with sequential node IDs
    remapped_subgraph = nx.Graph()
    for orig_node in relevant_nodes_list:
        new_node_id = node_mapping[orig_node]
        remapped_subgraph.add_node(new_node_id)

        # Copy node attributes if they exist
        if graph.has_node(orig_node):
            for attr_key, attr_val in graph.nodes[orig_node].items():
                remapped_subgraph.nodes[new_node_id][attr_key] = attr_val

    # Add edges with remapped node IDs
    for orig_u, orig_v in subgraph.edges():
        new_u = node_mapping[orig_u]
        new_v = node_mapping[orig_v]
        remapped_subgraph.add_edge(new_u, new_v)

        # Copy edge attributes if they exist
        if graph.has_edge(orig_u, orig_v):
            for attr_key, attr_val in graph.edges[orig_u, orig_v].items():
                remapped_subgraph.edges[new_u, new_v][attr_key] = attr_val

    # Create filtered event nodes matrix for subgraph
    event_nodes_sub = []
    for event_row in event_nodes:
        # Create new row with only relevant nodes
        new_row = [event_row[orig_id] if orig_id < len(event_row) else 0
                   for orig_id in relevant_nodes_list]
        event_nodes_sub.append(new_row)

    # Create filtered index event nodes for subgraph
    index_event_nodes_sub = {}
    for event_type, etb_list in index_event_nodes.items():
        if isinstance(etb_list, list):
            index_event_nodes_sub[event_type] = etb_list.copy()
        else:
            # Handle case where etb_list might be a single value
            index_event_nodes_sub[event_type] = etb_list

    # Create network data for subgraph (mapping node -> events produced)
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

    # Create all-pairs shortest path matrix for subgraph
    try:
        # Compute shortest paths for remapped subgraph
        all_pairs_dict = dict(nx.all_pairs_shortest_path_length(remapped_subgraph))

        # Convert to matrix format
        num_nodes = len(relevant_nodes_list)
        all_pairs_sub = [[float('inf')] * num_nodes for _ in range(num_nodes)]

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    all_pairs_sub[i][j] = 0
                elif j in all_pairs_dict.get(i, {}):
                    all_pairs_sub[i][j] = all_pairs_dict[i][j]

    except nx.NetworkXError:
        # Fallback: create basic distance matrix
        num_nodes = len(relevant_nodes_list)
        all_pairs_sub = [[1 if i != j else 0 for j in range(num_nodes)] for i in range(num_nodes)]

    # Create sub_network with same structure as network variable (list of Node objects)
    from Node import Node
    sub_network = []

    for new_node_id in sorted(reverse_mapping.keys()):
        orig_node_id = reverse_mapping[new_node_id]

        # Create new Node object for subgraph
        if orig_node_id < len(network):
            orig_node = network[orig_node_id]
            # Create new node with remapped ID but original attributes
            sub_node = Node(new_node_id, orig_node.computational_power, orig_node.memory)
            sub_node.eventrates = orig_node.eventrates.copy() if orig_node.eventrates else []

            # Remap parent/child/sibling relationships to subgraph node IDs
            sub_node.Parent = []
            sub_node.Child = []
            sub_node.Sibling = []

            # Only include relationships if both nodes are in the subgraph
            if hasattr(orig_node, 'Parent') and orig_node.Parent:
                for parent in orig_node.Parent:
                    if parent.id in node_mapping:
                        # Find the corresponding sub_node for this parent
                        parent_sub_id = node_mapping[parent.id]
                        # We'll need to set this after all nodes are created
                        sub_node.Parent.append(parent_sub_id)  # Store ID for now

            if hasattr(orig_node, 'Child') and orig_node.Child:
                for child in orig_node.Child:
                    if child.id in node_mapping:
                        child_sub_id = node_mapping[child.id]
                        sub_node.Child.append(child_sub_id)  # Store ID for now

            if hasattr(orig_node, 'Sibling') and orig_node.Sibling:
                for sibling in orig_node.Sibling:
                    if sibling.id in node_mapping:
                        sibling_sub_id = node_mapping[sibling.id]
                        sub_node.Sibling.append(sibling_sub_id)  # Store ID for now

            sub_network.append(sub_node)

    # Convert ID references back to object references
    for sub_node in sub_network:
        # Convert Parent IDs to Node objects
        parent_objects = []
        for parent_id in sub_node.Parent:
            if isinstance(parent_id, int):
                parent_node = next((n for n in sub_network if n.id == parent_id), None)
                if parent_node:
                    parent_objects.append(parent_node)
        sub_node.Parent = parent_objects

        # Convert Child IDs to Node objects
        child_objects = []
        for child_id in sub_node.Child:
            if isinstance(child_id, int):
                child_node = next((n for n in sub_network if n.id == child_id), None)
                if child_node:
                    child_objects.append(child_node)
        sub_node.Child = child_objects

        # Convert Sibling IDs to Node objects
        sibling_objects = []
        for sibling_id in sub_node.Sibling:
            if isinstance(sibling_id, int):
                sibling_node = next((n for n in sub_network if n.id == sibling_id), None)
                if sibling_node:
                    sibling_objects.append(sibling_node)
        sub_node.Sibling = sibling_objects

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


def calculate_prepp_costs_on_subgraph(self, node, subgraph, projection, central_eval_plan):
    """
    Calculate prepp costs on the extracted subgraph by generating evaluation plan and calling prePP.
    """
    from generateEvalPlan import generate_eval_plan
    from prepp import generate_prePP

    print("Calculate prepp on subgraph")

    # Create evaluation plan for subgraph
    subgraph_plan = central_eval_plan

    try:
        # Generate evaluation plan using existing function
        eval_plan_buffer = generate_eval_plan(
            nw=subgraph['sub_network'],
            selectivities=selectivities,
            myPlan=[subgraph_plan, {}, {}],  # Format: [plan, dict, dict]
            centralPlan=[0, {}, workload],  # Format: [source, dict, workload]
            workload=workload
        )

        # Create all_pairs matrix for subgraph (simplified distance matrix)
        all_pairs = subgraph['all_pairs_sub']

        # Call generate_prePP with the evaluation plan
        prepp_results = generate_prePP(
            input_buffer=eval_plan_buffer,
            method="ppmuse",
            algorithm="e",  # exact
            samples=1,
            top_k=1,
            runs=1,
            plan_print="f",  # false
            allPairs=all_pairs
        )

        # Extract costs from prePP results
        # prepp_results format: [exact_cost, pushPullTime, maxPushPullLatency, endTransmissionRatio]
        if prepp_results and len(prepp_results) > 0:
            costs = prepp_results[0]  # exact_cost
            print(f"PrePP costs calculated: {costs}")
            return costs
        else:
            print("Warning: No valid prePP results returned")
            return float('inf')

    except Exception as e:
        print(f"Error calculating prePP costs: {e}")
        # Fallback to simple cost calculation
        return _calculate_fallback_costs(node, subgraph, projection)


def _calculate_fallback_costs(node, subgraph, projection):
    """
    Fallback cost calculation if prePP fails.
    """
    try:
        # Simple cost based on distance from sources to sink
        total_cost = 0.0
        all_pairs = subgraph['all_pairs_sub']
        remapped_node = subgraph['placement_node_remapped']

        # Calculate basic routing costs
        for i in range(len(all_pairs)):
            if i != remapped_node:
                total_cost += all_pairs[remapped_node][i]

        print(f"Fallback costs calculated: {total_cost}")
        return total_cost

    except Exception as e:
        print(f"Error in fallback cost calculation: {e}")
        return float('inf')


def calculate_all_push_costs_on_subgraph():
    return 0.0  # Return 0 cost for now
