from typing import List, Any, Dict

from src.kraken2_0.components.event_stack_manager import update_event_stack
from src.kraken2_0.components.cost_calculator import CostCalculator
from src.kraken2_0.components.optimizer import PlacementOptimizer
from src.kraken2_0.components.sorter import EventPlacementSorter
from src.kraken2_0.state import SolutionCandidate, PlacementInfo


class PlacementProblem:
    def __init__(
        self,
        processing_order: List[Any],
        context: Dict[str, Any],
        enable_detailed_logging: bool = False,
    ):
        latency_threshold = context["latency_threshold"]
        """Initializes the entire problem context"""

        # Logging configuration
        self.logging_enabled = enable_detailed_logging
        if self.logging_enabled:
            self.detailed_log = []

        # --- Core Search & Problem Parameters ---
        self.processing_order = processing_order
        self.latency_threshold = (
            latency_threshold if latency_threshold is not None else float("inf")
        )
        self.query_workload = context["query_workload"]
        self.dependencies_per_projection = context["dependencies_per_projection"]
        self.latency_weighting_factor = context.get(
            "latency_weighting_factor", 1.0
        )  # The 'xi' factor

        # --- Instantiate Helper Component ---
        self.optimizer = PlacementOptimizer(
            context["graph"], context.get("routing_dict", {})
        )
        self.sorter = EventPlacementSorter()
        self.cost_calculator = CostCalculator(context)

        # Add problem reference to cost calculator for accessing placed subqueries
        self.cost_calculator.params["problem_ref"] = self

    def get_initial_candidate(self) -> SolutionCandidate:
        """Return the root node of the Solution Space S"""
        return SolutionCandidate()

    def is_goal(self, candidate: SolutionCandidate) -> bool:
        """Checks if a candidate represent a complete and valid solution e.g. a complete INEv graph."""

        # Here we should do more checks in the future, for now we only check if each projection has been placed.
        return len(candidate.placements) == len(self.processing_order)

    def expand(self, s_current: SolutionCandidate) -> List[SolutionCandidate]:
        """
        This function expands the current solution space with new states.

        The core "move generator". Given a state 's', it computes all valid next states by placing
        the next projection in the processing order and returns them.
        With respect to cost and latency constraints.
        """

        # Initialize the list holding the new states
        s_next_options = []

        # Pruning variable
        pruning = False

        # Get the projection index from the current state, this gives us the next projection to place.
        # If the current state is the start state, the index returns 0 starting with the first projection and so on
        projection_index = len(s_current.placements)

        if self.is_goal(s_current):
            return []

        # Get the next projection to be placed from the processing_order
        p = self.processing_order[projection_index]

        # Get and sort candidate physical nodes
        placed_subqueries = s_current.get_placed_subqueries(p, self)

        # (Pass necessary context to the optimizer)
        # This returns all the physical nodes n from the network T, that are capable of hosting this query simply based
        # on the question if all inputs are able to arrive there.
        possible_nodes = self.optimizer.get_possible_placement_nodes_optimized(
            p, placed_subqueries, self.cost_calculator.params
        )

        if not possible_nodes:
            return []

        # This sorts the nodes in an optimized manner to allow for pruning.
        # The goal is to: Check nodes that have all the inputs available first
        # Then go from the bottom of the network to the cloud, this way our latency constraint allows for pruning
        candidate_nodes = self.sorter.sort_candidate_nodes_optimized(
            possible_nodes, p, s_current.event_stack
        )

        # Now we can loop through each node n
        for n in candidate_nodes:
            # We calculate the costs for getting all the inputs to this node.
            # This returns a list of dicts, one for each valid comm strategy (e.g. push, predicate based push-pull)
            strategy_results = self.cost_calculator.calculate(p, n, s_current)

            # NOTE: For the pruning logic to work it is critical, that 'strategy_results' is order
            # with the lowest-latency strategy (all_push) appearing first.

            for result in strategy_results:
                # Here we need to create this next state to check its true latency
                s_next_temp = self._create_next_candidate(s_current, p, n, result)

                # Perform the accurate, step-by-step latency check.
                max_latency_so_far = s_next_temp.get_critical_path_latency(self)

                # Determine if this decision will be pruned
                is_pruned = False

                if (
                    result["strategy"] == "all_push"
                    and max_latency_so_far > self.latency_threshold
                ):
                    # Pruning condition achieved, we can prune everything.
                    pruning = True
                    is_pruned = True

                if max_latency_so_far > self.latency_threshold:
                    is_pruned = True

                # Log this placement decision if logging is enabled
                if self.logging_enabled:
                    log_entry = {
                        "projection_index": projection_index,
                        "projection": p,
                        "candidate_node": n,
                        "communication_strategy": result["strategy"],
                        "individual_cost": result["individual_cost"],
                        "transmission_latency": result["transmission_latency"],
                        "processing_latency": result["processing_latency"],
                        "cumulative_latency_so_far": max_latency_so_far,
                        "is_pruned": is_pruned,
                        "is_part_of_final_solution": False,  # Filled later
                    }
                    self.detailed_log.append(log_entry)

                if (
                    result["strategy"] == "all_push"
                    and max_latency_so_far > self.latency_threshold
                ):
                    break

                if max_latency_so_far <= self.latency_threshold:
                    s_next_options.append(s_next_temp)

            if pruning:
                break

        s_next_options.sort(key=lambda s: s.cumulative_cost)
        return s_next_options

    def _create_next_candidate(
        self, s_current: SolutionCandidate, p: Any, n: int, strategy_result: dict
    ) -> SolutionCandidate:
        """Helper function to create a new, extended SolutionCandidate object."""
        new_placements = s_current.placements.copy()
        new_event_stack = {k: v.copy() for k, v in s_current.event_stack.items()}

        placement_info = PlacementInfo(
            projection=p,
            node=n,
            strategy=strategy_result["strategy"],
            individual_cost=strategy_result["individual_cost"],
            individual_transmission_latency=strategy_result["transmission_latency"],
            individual_processing_latency=strategy_result["processing_latency"],
            acquisition_steps=strategy_result["acquisition_steps"],
        )

        new_placements[p] = placement_info

        # TODO: Implement this in /kraken/components/event_stack_manager.py
        update_event_stack(
            stack=new_event_stack, node_id=n, query_id=p, placement_info=placement_info
        )

        return SolutionCandidate(
            placements=new_placements,
            cumulative_cost=s_current.cumulative_cost + placement_info.individual_cost,
            event_stack=new_event_stack,
        )
