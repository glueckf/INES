#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:46:49 2021

@author: samira

Generate beneficial projections for given query workload.

"""
import helper.subsets as sbs
import multiprocessing
#from processCombination import *
# from filter import *
from helper.filter import getMaximalFilter, getDecomposedTotal
from helper.structures import getNumETBs, getNodes, getLongest
from helper.projString import filter_numbers, rename_without_numbers, getdoubles_k, parse_event_string
from helper.Tree import PrimEvent


def optimisticTotalRate(self, projection):  # USE FILTERED RATE FOR ESTIMATION
    rates = self.h_rates_data
    nodes = self.h_nodes
    projlist = self.h_projlist
    projFilterDict = self.h_projFilterDict
    IndexEventNodes = self.h_IndexEventNodes

    if projection in projlist:  # is complex event
        for i in projFilterDict.keys():
            if i == projection:
                myproj = i
                if getMaximalFilter(projFilterDict, myproj):
                    return getDecomposedTotal(getMaximalFilter(projFilterDict, myproj), myproj)
                else:
                    return projFilterDict[myproj][getMaximalFilter(projFilterDict, myproj)][0] * getNumETBs(myproj,
                                                                                                            IndexEventNodes)  #TODO change
    else:
        return rates[projection.leafs()[0]] * len(nodes[projection.leafs()[0]])


def optimisticTotalRate_single(self, projection):  # USE FILTERED RATE FOR ESTIMATION
    rates = self.h_rates_data
    nodes = self.h_nodes
    projrates = self.h_projrates
    projFilterDict = self.h_projFilterDict
    IndexEventNodes = self.h_IndexEventNodes
    for i in projFilterDict.keys():
        if i == projection:
            myproj = i
            if getMaximalFilter(projFilterDict, myproj):
                return getDecomposedTotal(getMaximalFilter(projFilterDict, myproj), myproj)
            else:
                return projrates[myproj][1] * getNumETBs(myproj, IndexEventNodes)  #TODO change
    else:
        #return 40
        return rates[projection.leafs()[0]] * len(nodes[projection.leafs()[0]])


def returnPartitioning(self, proj, combi, projrates: dict, *args):
    ''' returns list containing partitioning input type of proj generated with combi, args contains critical eventtypes, if potential eventtype in critical events, return False '''

    DistMatrices = None
    MSTrees = None
    rates = self.h_rates_data
    myevents = [x for x in combi if len(x) == 1]
    myevents = sorted(myevents, key=lambda x: rates[x], reverse=True)
    if args:
        args = args[0]
        if myevents:
            if myevents[0] in args:
                res, DistMatrices, MSTrees = NEW_isPartitioning(self, myevents[0], combi, proj, projrates)
                return [], DistMatrices, MSTrees  #,DistMatrices,MSTrees

    if myevents:
        res, DistMatrices, MSTrees = NEW_isPartitioning(self, myevents[0], combi, proj, projrates)
        #res = NEW_isPartitioning_alt(myevents[0], combi, proj, myprojFilterDict)
        if res:
            return [myevents[0], res[0]], DistMatrices, MSTrees
    return [], DistMatrices, MSTrees


def isPartitioning(self, element, combi, proj):
    ''' returns true if element partitioning input of proj generated with combi '''
    projrates = self.h_projrates
    rates = self.h_rates_data
    instances = self.h_instances
    IndexEventNodes = self.h_IndexEventNodes
    mysum = 0
    for i in combi:

        if i in rates.keys():
            additional = rates[i] * instances[i]
            mysum += additional

        else:
            additional = projrates[i][1] * getNumETBs(i, IndexEventNodes)
            mysum += additional  #  len(returnETBs(projection, network))

    mysum -= rates[element] * instances[element]
    mysum += projrates[proj][1] * getNumETBs(proj,
                                             IndexEventNodes)  # additional constraint about ratio of partitioning event type and outputrate of projection
    if rates[element] > mysum:
        return True

    else:
        return False


def isPartitioning_customRates(self, element, combi, proj, myrates):
    ''' returns true if element partitioning input of proj generated with combi '''
    rates = self.h_rates_data
    instances = self.h_instances
    IndexEventNodes = self.h_IndexEventNodes

    mysum = 0
    for i in combi:

        if i in rates.keys():
            additional = rates[i] * instances[i]
            mysum += additional

        else:
            additional = myrates[i] * getNumETBs(i, IndexEventNodes)
            mysum += additional  #  len(returnETBs(projection, network))
    mysum -= rates[element] * instances[element]
    mysum += myrates[proj] * getNumETBs(proj,
                                        IndexEventNodes)  # additional constraint about ratio of partitioning event type and outputrate of projection
    if rates[element] > mysum:
        return True

    else:
        return False


""" 
def NEW_isPartitioning_customRates(self, element, combi, proj, myrates):
		''' returns true if element partitioning input of proj generated with combi '''
		from networkx.algorithms.approximation import steiner_tree

		rates = self.h_rates_data
		nodes = self.h_nodes
		longestPath = getLongest(self.allPairs)
		G = self.graph
		wl = self.query_workload
		projrates = self.h_projrates
		IndexEventNodes = self.h_IndexEventNodes
		etbs = IndexEventNodes[element]
		myNodes = [getNodes(x)[0] for x in etbs]   
		if element not in MSTrees.keys():
			myTree = steiner_tree(G, myNodes)
			MSTrees[element] = myTree
		else:
			myTree =  MSTrees[element]
		
		if myTree not in DistMatrices.keys():           
			myAllPairs = fillMyDistMatrice(myTree)
			DistMatrices[myTree] = myAllPairs
		else:
			myAllPairs = DistMatrices[myTree]       
		
		#bestNodeValue = min([sum(x) for x in myAllPairs if myAllPairs and x])           
		costs = len(myTree.edges())                
		
		mysum =  0    
		for i in [x for x in combi if not x == element]:            
			if i in rates.keys():        
				additional = rates[i] * len(nodes[i])              
				mysum += additional              
			else:
				additional = myrates[i] * getNumETBs(i)
				mysum += additional

		myproj  = 0 #costs for outputrates
		if proj.get_original(wl) not in wl:
			myproj =   myrates[proj] * (getNumETBs(proj)) # additional constraint about ratio of partitioning event type and outputrate of projection

		if totalRate(element,projrates) * longestPath > (mysum * costs) + myproj * longestPath :  
			
			return [costs]

		else: 
			return False  """


def minimum_subgraph(G, nodes_list):
    # Initialize an empty set to hold the edges in the minimum subgraph
    import networkx as nx
    subgraph_edges = set()

    # For each pair of nodes in the list, find the shortest path
    for i, source in enumerate(nodes_list):
        for target in nodes_list[i + 1:]:
            try:
                # Get the shortest path as a list of nodes
                path = nx.shortest_path(G, source=source, target=target, method='dijkstra')

                # Add all the edges in this path to the subgraph
                subgraph_edges.update(zip(path[:-1], path[1:]))

            except nx.NetworkXNoPath:
                # If no direct path exists, connect via node 0 (the cloud node)
                try:
                    # Get shortest path from source to node 0
                    path_source_to_0 = nx.shortest_path(G, source=source, target=0, method='dijkstra')
                    # Get shortest path from target to node 0
                    path_target_to_0 = nx.shortest_path(G, source=target, target=0, method='dijkstra')

                    # Combine paths via node 0 (source -> 0 -> target)
                    # Add source to 0 edges
                    subgraph_edges.update(zip(path_source_to_0[:-1], path_source_to_0[1:]))
                    # Add target to 0 edges in reverse direction (node 0 -> target)
                    subgraph_edges.update(zip(path_target_to_0[:-1], path_target_to_0[1:]))

                except nx.NetworkXNoPath:
                    # If there's no path to node 0, raise an exception or handle as needed
                    print(
                        f"[PROJECTION] Warning: Neither node {source} nor {target} has a path to cloud node 0 - topology may be invalid")

    # Create the subgraph from the collected edges
    subgraph = G.edge_subgraph(subgraph_edges).copy()

    return subgraph


def NEW_isPartitioning(self, element, combi, proj, projrates: dict):
    ''' returns true if element partitioning input of proj generated with combi '''
    MSTrees = {}
    DistMatrices = {}
    rates = self.h_rates_data
    nodes = self.h_nodes
    wl = self.query_workload
    G = self.graph
    IndexEventNodes = self.h_IndexEventNodes
    EventNodes = self.h_eventNodes

    longestPath = getLongest(self.allPairs)

    etbs = IndexEventNodes[element]
    myNodes = [getNodes(x, EventNodes, IndexEventNodes)[0] for x in etbs]
    if element not in MSTrees.keys():
        myTree = minimum_subgraph(G, myNodes)
        MSTrees[element] = myTree
    else:
        myTree = MSTrees[element]

    if myTree not in DistMatrices.keys():
        myAllPairs = fillMyDistMatrice(myTree)
        DistMatrices[myTree] = myAllPairs
    else:
        myAllPairs = DistMatrices[myTree]

    #bestNodeValue = min([sum(x) for x in myAllPairs if myAllPairs and x])
    costs = len(myTree.edges())

    mysum = 0
    for i in [x for x in combi if not x == element]:
        if i in rates.keys():
            additional = rates[i] * len(nodes[i])
            mysum += additional
        else:
            additional = projrates[i][1] * getNumETBs(i, IndexEventNodes)
            mysum += additional

    myproj = 0  #costs for outputrates
    if proj.get_original(wl) not in wl:
        myproj = projrates[proj][1] * (getNumETBs(proj,
                                                  IndexEventNodes))  # additional constraint about ratio of partitioning event type and outputrate of projection

    if totalRate(self, element, projrates) * longestPath > (mysum * costs) + myproj * longestPath:

        return [costs], MSTrees, DistMatrices

    else:
        return False, MSTrees, DistMatrices


def fillMyMatrice(myNodes, myEdges, me):
    import networkx as nx
    myG = nx.Graph()
    myG.add_nodes_from(myNodes)
    myG.add_edges_from(myEdges)
    myDistances = []
    for j in myNodes:
        myDistances.append(len(nx.shortest_path(myG, me, j, method='dijkstra')) - 1)
    return (me, myDistances)


pool = None  # Only declare the pool variable once


def get_pool():
    global pool
    if pool is None:
        pool = multiprocessing.Pool()
    return pool


def fillMyDistMatrice(myG):  #all pairs shortest path distance matrice -> also slow for big graphs
    global pool
    myNodes = list(myG.nodes)
    myPairs = [[] for x in myNodes]
    mytuple = (list(myNodes), list(myG.edges))
    args = [(mytuple[0], mytuple[1], x) for x in myNodes]
    # Use the pool to parallelize the computation
    current_pool = get_pool()
    result = current_pool.starmap(fillMyMatrice, args)

    for i in result:
        myPairs.append(i[1])
    return myPairs


def min_max_doubles(query, projevents):
    doubles = getdoubles_k(projevents)
    leafs = map(lambda x: filter_numbers(x), query.leafs())
    for event in doubles.keys():
        if not doubles[event] == leafs.count(event):
            return False
    return True


def settoproj(evlist, query):
    """ take query and list of prim events and return projection"""

    leaflist = []
    evlist = parse_event_string(evlist)
    evlist = list(map(lambda x: str(x), evlist))
    for i in evlist:
        leaflist.append(PrimEvent(i))
    newproj = query.getsubop(leaflist)
    return newproj


def isBeneficial(self, projection, rate):
    """ determines for a projection based on the if it is beneficial """
    rates = self.h_rates_data
    nodes = self.h_nodes
    totalProjrate = rate * getNumETBs(projection, self.h_IndexEventNodes)
    sumrates = sum(map(lambda x: rates[x] * float(len(nodes[x])), projection.leafs()))
    if sumrates > totalProjrate:
        return True
    else:
        return False


def totalRate(self, projection, projrates: dict):
    rates = self.h_rates_data
    nodes = self.h_nodes
    IndexEventNodes = self.h_IndexEventNodes

    if projection in projrates.keys():  # is complex event
        return projrates[projection][1] * getNumETBs(projection, IndexEventNodes)

    elif isinstance(projection, str) or len(projection) == 1:
        if isinstance(projection, str):
            proj_key = projection
        else:
            proj_key = str(projection)

        return rates[proj_key] * len(nodes[proj_key])
    else:
        outrate = projection.evaluate() * getNumETBs(projection, IndexEventNodes)
        selectivity = return_selectivity(self, projection.leafs())
        myrate = outrate * selectivity
        return myrate


def return_selectivity(self, proj):
    """ return selectivity for arbitrary projection """
    if not hasattr(self, 'selectivities'):
        # If using the projection directly, not through INES
        return 1

    selectivities = self.selectivities
    proj = list(map(lambda x: filter_numbers(x), proj))
    two_temp = sbs.printcombination(proj, 2)
    selectivity = 1
    for two_s in two_temp:
        if two_s in selectivities.keys():
            if selectivities[two_s] != 1:
                selectivity *= selectivities[two_s]
    return selectivity


def generate_projections(self, query):
    """ generates list of benecifical projection """
    negated = query.get_negated()
    projections = []
    projrates = {}
    match = query.leafs()
    projlist = match
    for i in range(2, len(match)):
        iset = sbs.boah(match, i)
        for k in range(len(iset)):
            nseq_violated = False
            curcom = list(iset[k].split(","))
            projevents = rename_without_numbers(
                "".join(sorted(list(set(curcom)))))  #A1BC becomes ABC and A1B1CA2 becomes A1BCA2
            mysubop = settoproj(curcom, query)
            #mysubop = mysubop.rename_leafs(sepnumbers(projevents)) #renaming on tree > A1BC becomes ABC and A1B1CA2 becomes A1BCA2
            for neg in negated:  #if negated type in projection
                if neg in mysubop.getleafs():
                    mycontext = query.get_context(neg)
                    if not set(mycontext).issubset(
                            set(mysubop.getleafs())):  # if conext of negated event not in projection, exclude projection
                        nseq_violated = True
            outrate = mysubop.evaluate(self.h_rates_data)

            #outrate = mysubop.evaluate(self.h_rates_data)
            selectivity = self.return_selectivity(curcom)
            rate = outrate * selectivity
            placement_options = isBeneficial(self, mysubop, rate)

            if placement_options and min_max_doubles(query,
                                                     projevents) and not nseq_violated:  # if the projection is beneficial (yields a placement option) and minmax?
                projrates[mysubop] = (selectivity, rate)
                projections.append(mysubop)  # do something to prevent a1a2b and a2a3b to be appended to dictionary
    projections.append(query)
    outrate = query.evaluate(self.h_rates_data)
    selectivity = self.return_selectivity(query.leafs())
    rate = outrate * selectivity
    projrates[query] = (selectivity, rate)
    # Debug output for projection generation
    # print(f"[PROJECTION] Generated {len(projections)} projections")
    # print(f"[PROJECTION] Projection rates: {projrates}")
    return projections, projrates


def returnSubProjections(proj, projlist):
    """ return list of projection keys that can be used in a combination of a given projection"""
    myprojlist = [x for x in projlist if
                  len(x.leafs()) <= len(proj.leafs()) and set(x.leafs()).issubset(set(proj.leafs()))]
    outputlist = []
    for i in myprojlist:
        if not proj == i:
            if i.can_be_used(proj):
                outputlist.append(i)

    return outputlist


#_sharedProjectionsDict = {}
#_sharedProjectionsList = []
#_projsPerQuery = {}
#_projlist = []
#_projrates = {}

def generate_all_projections(self):
    #"TODO no global"
    sharedProjectionsDict = self.h_sharedProjectionsDict
    sharedProjectionsList = self.h_sharedProjectionsList
    projsPerQuery = self.h_projsPerQuery
    projlist = self.h_projlist
    projrates = self.h_projrates

    wl = self.query_workload
    for query in wl:

        query = query.stripKL_simple()
        result = generate_projections(self, query)

        #projsPerQuery[query] = result[0]
        for i in result[0]:
            if i not in projlist:
                projlist.append(i)
                projrates[i] = result[1][i]
                sharedProjectionsDict[i] = [query]
            else:
                for mykey in sharedProjectionsDict.keys():
                    if mykey == i:
                        sharedProjectionsDict[mykey].append(query)
    #print(projrates)
    for query in wl:
        query = query.stripKL_simple()
        projsPerQuery[query] = [x for x in projlist if query.can_be_used(x)]

    for projection in sharedProjectionsDict.keys():
        if len(sharedProjectionsDict[projection]) > 1:
            sharedProjectionsList.append(projection)

    return projlist, projrates, projsPerQuery, sharedProjectionsDict, sharedProjectionsList


if __name__ == '__main__':
    # Initialize the multiprocessing pool
    from multiprocessing import freeze_support

    freeze_support()
    pool = multiprocessing.Pool()
