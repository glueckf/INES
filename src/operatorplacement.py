from helper.placement_aug import NEWcomputeCentralCosts, ComputeSingleSinkPlacement, computeMSplacementCosts
from helper.processCombination_aug import compute_dependencies, getSharedMSinput
import time
import csv
import sys
import argparse
from EvaluationPlan import EvaluationPlan
import numpy as np
from projections import returnPartitioning, totalRate
from placement_engine.core import compute_operator_placement_with_prepp
from placement_engine.global_placement_tracker import get_global_placement_tracker, reset_global_placement_tracker


#maxDist = max([max(x) for x in allPairs])

def getLowerBound(query,
                  self):  # lower bound -> for multiple projections, keep track of events sent as single sink and do not add up

    projsPerQuery = self.h_projsPerQuery
    rates = self.h_rates_data
    longestPath = self.h_longestPath

    MS = []
    for e in query.leafs():
        # myprojs = [p for p in list(set(projsPerQuery[query]).difference(set([query]))) if
        #            totalRate(p) < rates[e] and not e in p.leafs()]

        # Step 1: Get projects for this query
        projects_for_query = projsPerQuery[query]

        # Step 2: Remove duplicates and convert to set
        projects_set = set(projects_for_query)

        # Step 3: Remove the query itself from the projects
        query_set = set([query])
        filtered_projects_set = projects_set.difference(query_set)

        # Step 4: Convert back to list
        filtered_projects_list = list(filtered_projects_set)

        # Step 5: Filter by conditions
        myprojs = []

        for p in filtered_projects_list:
            if totalRate(self, p, self.h_projrates) < rates[e] and not e in p.leafs():
                myprojs.append(p)
        if myprojs:
            MS.append(e)
        for p in [x for x in projsPerQuery[query] if e in x.leafs()]:
            part = returnPartitioning(self, p, p.leafs(), self.h_projrates, self.h_combiDict)

            if e in part:
                MS.append(e)
    nonMS = [e for e in query.leafs() if not e in MS]
    if nonMS:
        minimalRate = sum(
            sorted([totalRate(self, e, self.h_projrates) for e in query.leafs() if not e in MS])) * longestPath
    else:
        minimalRate = min([totalRate(self, e, self.h_projrates) for e in query.leafs()]) * longestPath
    minimalProjs = sorted([totalRate(self, p, self.h_projrates) for p in projsPerQuery[query] if not p == query])[
                   :len(list(set(MS))) - 1]
    if not len(nonMS) == len(query.leafs()):
        minimalRate += sum(minimalProjs) * longestPath

    # print("→ totalRate call for:", self.h_projection)
    # print("→ result:", self.h_projrates.get(self.h_projection, 0))

    return minimalRate  #, nonMS)


def calculate_operatorPlacement(self, file_path: str, max_parents: int):
    wl = self.query_workload
    allPairs = self.allPairs
    #getNetworkParameters, selectivityParameters, combigenParameters
    networkParams = self.networkParams
    selectivityParams = self.selectivitiesExperimentData
    combigenParams = self.h_combiExperimentData
    longestPath = self.h_longestPath
    projFilterDict = self.h_projFilterDict
    IndexEventNodes = self.h_IndexEventNodes
    allPairs = self.allPairs
    rates = self.h_rates_data
    network = self.network
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
    print(f"[PLACEMENT] Index Event Nodes: {IndexEventNodes}")
    ccosts = NEWcomputeCentralCosts(wl, IndexEventNodes, allPairs, rates, EventNodes, self.graph)
    #print("central costs : " + str(ccosts))
    centralHopLatency = max(allPairs[ccosts[1]])
    numberHops = sum(allPairs[ccosts[1]])
    print(f"[CENTRAL COSTS] Cost: {ccosts[0]:.2f}")
    print(f"[CENTRAL COSTS] Total Hops: {numberHops}")
    print(f"[CENTRAL COSTS] Hop Latency: {centralHopLatency:.2f}")
    MSPlacements = {}
    curcosts = 1
    start_time = time.time()

    # Initialize global placement tracker for this placement session
    reset_global_placement_tracker()  # Start fresh for each placement calculation
    global_tracker = get_global_placement_tracker()
    print(f"[PLACEMENT] Initialized global placement tracker")

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
    # print(f"[MS PLACEMENT] Unfolded projections: {unfolded}")
    # print(f"[MS PLACEMENT] Critical MS types: {criticalMSTypes}")
    dependencies = compute_dependencies(self, unfolded, criticalMSTypes)
    processingOrder = sorted(dependencies.keys(), key=lambda x: dependencies[x])  # unfolded enthält kombi
    costs = 0

    central_eval_plan = [ccosts[1], ccosts[3], wl]

    temp_results_dict = {}

    for projection in processingOrder:  #parallelize computation for all projections at the same level
        if set(unfolded[projection]) == set(projection.leafs()):  #initialize hop latency with maximum of children
            hopLatency[projection] = 0
        else:
            hopLatency[projection] = max([hopLatency[x] for x in unfolded[projection] if x in hopLatency.keys()])

        #partType = returnPartitioning(self,projection, unfolded[projection], self.h_projrates,criticalMSTypes)

        # ComputeMSPlacement
        # TODO: Currntly leave out MS placement for integrated approach, as it is not yet implemented
        # partType,_,_ = returnPartitioning(self, projection, unfolded[projection], projrates ,criticalMSTypes)
        partType = False
        if partType:
            MSPlacements[projection] = partType
            result = computeMSplacementCosts(self, projection, unfolded[projection], partType, sharedDict, noFilter, G)
            # if not result:
            #     print(f"[Fehler] Leeres Ergebnis für MS-Placement von {projection} erhalten. Überspringe...")
            #     continue  # continue / return / break
            additional = result[0]
            #     print(f"[DEBUG] MS Projection: {projection}")
            #     print(f"        → PlacementCost: {additional}")
            #     print(f"        → Sink: {result[2].sinks}")
            #    #print(f"        → Used Nodes: {[inst.nodes for inst in result[3]]}")
            #     print(f"        → Routes: {[inst.routingDict.get(projection) for inst in result[3] if projection in inst.routingDict]}")

            costs += additional
            hopLatency[projection] += result[1]

            myPlan.addProjection(result[2])  #!

            for newin in result[2].spawnedInstances:  # add new spawned instances
                myPlan.addInstances(projection, newin)

            myPlan.updateInstances(result[3])  #! update instances

            Filters += result[4]
            #if partType, and projection in wl and partType kleene component of projection, add sink
            # print(f"[MS PLACEMENT] {projection} → Node: {partType}, Cost: {additional:.2f}, Hops: {result[1]}")

            if projection.get_original(wl) in wl and partType[0] in list(
                    map(lambda x: str(x), projection.get_original(wl).kleene_components())):
                result = ComputeSingleSinkPlacement(projection.get_original(wl), [projection], noFilter)
                additional = result[0]
                costs += additional
                # print(f"[SiS KLEENE] Sink at {partType}{projection.get_original(wl)}, Cost: {additional:.2f}, Hops: {result[1]}")

        else:
            # INFO: ComputeSingleSinkPlacement is called for the sequential approach.
            # Implementing a new function for the integrated approach
            # if str(projection) == 'AND(SEQ(B, C), F)':
            #     print("Hook")

            # result = ComputeSingleSinkPlacement(projection, unfolded[projection], noFilter, projFilterDict, EventNodes,
            #                                     IndexEventNodes, self.h_network_data, allPairs, mycombi, rates,
            #                                     singleSelectivities, projrates, self.graph, self.network)

            result = compute_operator_placement_with_prepp(
                self,
                projection,
                unfolded[projection],
                noFilter,
                projFilterDict,
                EventNodes,
                IndexEventNodes,
                self.h_network_data,
                allPairs, mycombi,
                rates,
                singleSelectivities,
                projrates,
                G,
                network,
                central_eval_plan)

            temp_results_dict[projection] = result
            print(result)

            # additional = result[0]
            # costs += additional
            # hopLatency[projection] += result[2]
            # myPlan.addProjection(result[3])  #!
            # for newin in result[3].spawnedInstances:  # add new spawned instances
            #
            #     myPlan.addInstances(projection, newin)
            #
            # myPlan.updateInstances(result[4])  #! update instances
            # Filters += result[5]


            # print(f"[SiS PLACEMENT] {projection} → Node: {partType}, Cost: {additional:.2f}, Hops: {result[2]}")
    print(temp_results_dict)

    total_costs = 0
    for projection in temp_results_dict:
        if projection in self.query_workload:
            total_costs += temp_results_dict[projection].costs


    # Print global placement tracker summary
    print(f"\n[PLACEMENT] Global Placement Tracker Summary:")
    print(global_tracker.get_summary())
    
    mycosts = costs / ccosts[0]
    print(f"[TRANSMISSION] INES with MS - Total Cost: {costs:.2f}")
    if len(wl) > 1 or wl[0].hasKleene() or wl[0].hasNegation():
        lowerBound = 0
    else:
        for query in wl:
            lowerBound = getLowerBound(query, self)
    print(f"[BOUNDS] Lower Bound Ratio: {lowerBound / ccosts[0]:.4f}")

    print(f"[TRANSMISSION] Ratio: {mycosts:.4f}")
    #print("INEv Depth: " + str(float(max(list(dependencies.values()))+1)/2))

    ID = int(np.random.uniform(0, 10000000))

    totaltime = str(round(time.time() - start_time, 2))

    print(f"[TIMING] Execution Summary:")
    print(f"[TIMING] Start: {start_time:.2f}")
    print(f"[TIMING] End: {time.time():.2f}")
    print(f"[TIMING] Duration: {totaltime} seconds")

    ID = int(np.random.uniform(0, 10000000))

    print(f"[DEPENDENCIES] Final dependencies: {dependencies}")
    #hoplatency = max([hopLatency[x] for x in hopLatency.keys()])   
    if dependencies:
        max_dependency = float(max(list(dependencies.values())) / 2)
    else:
        max_dependency = 0.0  # default value
    #totalLatencyRatio = hoplatency / centralHopLatency
    myResult = [ID, mycosts, ccosts[0], costs, Filters, networkParams[3], networkParams[0], networkParams[2], len(wl),
                combigenParams[3], selectivityParams[0], selectivityParams[1], combigenParams[1], longestPath,
                totaltime, centralHopLatency, max_dependency, ccosts[0], lowerBound / ccosts[0], networkParams[1],
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
    experiment_result = [ID, costs]
    return eval_Plan, central_eval_plan, experiment_result, myResult
