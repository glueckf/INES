import networkx

from src.placement_engine.state import PlacementDecisionTracker


def update_placement_state_with_best_decision(
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
        central_eval_plan: list,
        placement_decisions: PlacementDecisionTracker,
        sinks=None
) -> None:

    """
    For now, keep this function as it is.
    Later on we need to evaluate the best decisions with respect to the previous placement and possible future placements.


    """
    return None




