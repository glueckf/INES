from helper.placement_aug import NEWcomputeCentralCosts, ComputeSingleSinkPlacement, computeMSplacementCosts
from helper.processCombination_aug import compute_dependencies, getSharedMSinput
import time
import csv
import sys
import argparse
from EvaluationPlan import EvaluationPlan
import numpy as np
from projections import returnPartitioning, totalRate


#maxDist = max([max(x) for x in allPairs])

def getLowerBound(query,
                  self):  # lower bound -> for multiple projections, keep track of events sent as single sink and do not add up

    projsPerQuery = self.h_projsPerQuery
    rates = self.h_rates_data
    longestPath = self.h_longestPath

    MS = []
    for e in query.leafs():
        myprojs = [p for p in list(set(projsPerQuery[query]).difference(set([query]))) if
                   totalRate(p) < rates[e] and not e in p.leafs()]
        if myprojs:
            MS.append(e)
        for p in [x for x in projsPerQuery[query] if e in x.leafs()]:
            part = returnPartitioning(p, p.leafs())
            if e in part:
                MS.append(e)
    nonMS = [e for e in query.leafs() if not e in MS]
    if nonMS:
        minimalRate = sum(sorted([totalRate(e) for e in query.leafs() if not e in MS])) * longestPath
    else:
        minimalRate = min([totalRate(e) for e in query.leafs()]) * longestPath
    minimalProjs = sorted([totalRate(p) for p in projsPerQuery[query] if not p == query])[:len(list(set(MS))) - 1]
    if not len(nonMS) == len(query.leafs()):
        minimalRate += sum(minimalProjs) * longestPath
    return minimalRate  #, nonMS)


def calculate_operatorPlacement(self, file_path: str, max_parents: int):

    query_workload = self.query_workload
    networkParams = self.networkParams
    selectivityParams = self.selectivitiesExperimentData
    combigenParams = self.h_combiExperimentData
    longestPath = self.h_longestPath
    projFilterDict = self.h_projFilterDict
    IndexEventNodes = self.h_IndexEventNodes
    allPairs = self.allPairs
    rates = self.h_rates_data
    mycombi = self.h_mycombi
    singleSelectivities = self.single_selectivity
    projrates = self.h_projrates
    EventNodes = self.h_eventNodes
    G = self.graph

    Filters = []
    writeExperimentData = 0

    filename = "results"
    noFilter = 0  # NO FILTER

    # Access the arguments
    filename = file_path
    number_parents = max_parents

    print(f"[PLACEMENT] Processing file: {filename}")
    print(f"[PLACEMENT] Analyzing placement options...")
    print(f"[PLACEMENT] IndexEventNodes structure: {IndexEventNodes}")
    ccosts = NEWcomputeCentralCosts(query_workload, IndexEventNodes, allPairs, rates, EventNodes, self.graph)
    #print("central costs : " + str(ccosts))
    centralHopLatency = max(allPairs[ccosts[1]])
    numberHops = sum(allPairs[ccosts[1]])
    print(f"[PLACEMENT] Central processing costs: {ccosts[0]}")
    print(f"[PLACEMENT] Central communication hops: {numberHops}")
    print(f"[PLACEMENT] Central hop latency: {centralHopLatency}")
    MSPlacements = {}
    curcosts = 1
    start_time = time.time()

    hopLatency = {}

    #Reduce calls of initEventNodes
    #init_eventNodes = initEventNodes()   
    EventNodes = self.h_eventNodes
    IndexEventNodes = self.h_IndexEventNodes

    myPlan = EvaluationPlan([], [])

    #transforming indexeventnodes into EvaluationPLan object with all entries as a instance
    #jede instance ist eine node ein event (nodes * events die produziert werden pro node)
    myPlan.initInstances(IndexEventNodes)  # init with instances for primitive event types

    #mycombi = removeSisChains()
    unfolded = self.h_mycombi
    criticalMSTypes = self.h_criticalMSTypes
    sharedDict = getSharedMSinput(self, unfolded, projFilterDict)
    print(f"[PLACEMENT] Unfolded projection combinations: {unfolded}")
    print(f"[PLACEMENT] Critical multi-sink types: {criticalMSTypes}")
    dependencies = compute_dependencies(self, unfolded, criticalMSTypes)
    processingOrder = sorted(dependencies.keys(), key=lambda x: dependencies[x])  # unfolded enthält kombi
    costs = 0

    for projection in processingOrder:  #parallelize computation for all projections at the same level
        if set(unfolded[projection]) == set(projection.leafs()):  #initialize hop latency with maximum of children
            hopLatency[projection] = 0
        else:
            hopLatency[projection] = max([hopLatency[x] for x in unfolded[projection] if x in hopLatency.keys()])

        partType, _, _ = returnPartitioning(self, projection, unfolded[projection], projrates, criticalMSTypes)
        if partType:
            MSPlacements[projection] = partType
            result = computeMSplacementCosts(self, projection, unfolded[projection], partType, sharedDict, noFilter, G)
            if not result:
                print(f"[PLACEMENT] Error: Empty result for MS-placement of {projection}. Skipping...")
                continue  # continue / return / break
            additional = result[0]
            costs += additional
            hopLatency[projection] += result[1]

            myPlan.addProjection(result[2])  #!

            for newin in result[2].spawnedInstances:  # add new spawned instances
                myPlan.addInstances(projection, newin)

            myPlan.updateInstances(result[3])  #! update instances

            Filters += result[4]
            #if partType, and projection in query_workload and partType kleene component of projection, add sink
            print(
                f"[PLACEMENT] Multi-sink placement - Projection: {projection}, Location: {partType}, Cost: {additional}, Hops: {result[1]}")

            if projection.get_original(query_workload) in query_workload and partType[0] in list(
                    map(lambda x: str(x), projection.get_original(query_workload).kleene_components())):
                result = ComputeSingleSinkPlacement(projection.get_original(query_workload), [projection], noFilter)
                additional = result[0]
                costs += additional
                print(
                    f"[PLACEMENT] Single-sink placement for Kleene - Location: {partType}, Query: {projection.get_original(query_workload)}, Cost: {additional}, Hops: {result[1]}")

        else:
            result = ComputeSingleSinkPlacement(projection, unfolded[projection], noFilter, projFilterDict, EventNodes,
                                                IndexEventNodes, self.h_network_data, allPairs, mycombi, rates,
                                                singleSelectivities, projrates, self.graph, self.network)
            additional = result[0]
            costs += additional
            hopLatency[projection] += result[2]
            myPlan.addProjection(result[3])  #!
            for newin in result[3].spawnedInstances:  # add new spawned instances

                myPlan.addInstances(projection, newin)

            myPlan.updateInstances(result[4])  #! update instances
            Filters += result[5]

            print(
                f"[PLACEMENT] Single-sink placement - Projection: {projection}, Location: {partType}, Cost: {additional}, Hops: {result[2]}")

    mycosts = costs / ccosts[0]
    print(f"[PLACEMENT] INES transmission cost with multi-sink optimizations: {costs}")
    if len(query_workload) > 1 or query_workload[0].hasKleene() or query_workload[0].hasNegation():
        lowerBound = 0
    else:
        for query in query_workload:
            lowerBound = getLowerBound(query, self)
    print(f"[PLACEMENT] Theoretical lower bound ratio: {lowerBound / ccosts[0]:.3f}")

    print(f"[PLACEMENT] Achieved transmission ratio: {mycosts:.3f}")
    #print("INEv Depth: " + str(float(max(list(dependencies.values()))+1)/2))

    ID = int(np.random.uniform(0, 10000000))

    totaltime = str(round(time.time() - start_time, 2))

    print(f"[PLACEMENT] Operator placement computation complete")
    print(f"[PLACEMENT] Start time: {start_time}")
    print(f"[PLACEMENT] End time: {time.time()}")
    print(f"[PLACEMENT] Total execution time: {totaltime} seconds")

    ID = int(np.random.uniform(0, 10000000))

    print(f"[PLACEMENT] Projection dependencies: {dependencies}")
    #hoplatency = max([hopLatency[x] for x in hopLatency.keys()])   
    if dependencies:
        max_dependency = float(max(list(dependencies.values())) / 2)
    else:
        max_dependency = 0.0  # default value

    myResult = [
        ID,
        mycosts,
        ccosts[0],
        costs,
        Filters,
        networkParams["network_size"],
        networkParams["eventskew"],
        networkParams["node_event_ratio"],
        len(query_workload),
        combigenParams[3],
        selectivityParams[0],
        selectivityParams[1],
        combigenParams[1],
        longestPath,
        totaltime,
        centralHopLatency,
        max_dependency,
        ccosts[0],
        lowerBound / ccosts[0],
        networkParams["number_eventtypes"],
        number_parents]

    # new = False
    # try:
    #      f = open("./res/"+str(filename)+".csv")   
    # except FileNotFoundError:
    #      new = True           

    # with open("./res/"+str(filename)+".csv", "a") as result:
    #    writer = csv.writer(result)  
    #    if new:
    #        writer.writerow(schema)              
    #    writer.writerow(myResult)
    #with open('EvaluationPlan',  'wb') as EvaluationPlan_file:
    # pickle.dump([myPlan, ID, MSPlacements], EvaluationPlan_file)
    eval_Plan = [myPlan, ID, MSPlacements]
    central_eval_plan = [ccosts[1], ccosts[3], query_workload]
    experiment_result = [ID, costs]
    return eval_Plan, central_eval_plan, experiment_result, myResult
