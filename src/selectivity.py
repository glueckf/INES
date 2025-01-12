"""
Initialize selectivities for given tuple of primitive event types (projlist) within interval [x,y].
"""
import random as rd
import numpy as np
from helper.projString import generate_twosets,changeorder
from queryworkload import get_primitive_events


def initialize_selectivities(primEvents,x=0.1,y=0.01): 

    primEvents = get_primitive_events(primEvents)
   
    projlist = generate_twosets(primEvents)       
    projlist = list(set(projlist))
    selectivities = {}
    selectivity = 0
    for i in projlist: 
        #if len(filter_numbers(i)) >1 :                  
            selectivity = rd.uniform(0.0,0.3)             
            if selectivity > 0.2:
                selectivity = 1
                selectivities[str(i)] =  selectivity
                selectivities[str(changeorder(i))] =  selectivity
            if selectivity < 0.2: 
                selectivity = rd.uniform(x,y)                
                selectivities[str(i)] =  selectivity
                selectivities[str(changeorder(i))] =  selectivity
    selectivitiesExperimentData = [x, np.median(list(selectivities.values()))]
    return selectivities,selectivitiesExperimentData



    
def increase_selectivities(percentage, selectivities): # multiply x percent of selectivities with factor ? -> 10 for starters
    number =int( len(list(selectivities.keys())) / 100) * percentage
    mylist = list(selectivities.keys())
    rd.shuffle(mylist)
    projs = mylist[0:number]
    for proj in projs:
        selectivities[proj] = selectivities[proj]  * 10 
    return selectivities