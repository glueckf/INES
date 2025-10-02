from typing import Any, Dict, List

from src.kraken2_0.state import SolutionCandidate


class CostCalculator:
    """
    A modular component responsible for the entire cost and latency calculation pipeline.

    It encepsulates the logic for calling the PrePP function, adjusting for local events
    and adding the final sink costs.
    """
    def __init__(self, context):
        self.context = context

    def calculate(self, p: Any, n: int, s_current: SolutionCandidate) -> List[Dict[str, Any]]:
        """
        The main public method. Orchestrates the full cost calculation pipeline.

        Returns:
            A list of dictionaries, where each dictionary represents a valid cost and latency calculation
            for a communication strategy (e.g., "all_push", "push_pull") and its details.
        """

        raw_results = self._get_raw_costs_from_prepp(p, n, s_current)

        if not raw_results:
            return[]

        adjusted_results = self._adjust_for_local_events(raw_results, n , s_current)

        final_results = self._add_sink_costs(adjusted_results, p, n)

        return self._format_results(final_results)

    def _get_raw_costs_from_prepp(self, p, n, s_current):
        # Here we need to implement the current logic from the calculate_prepp_with_placement
        # Preparing the input buffer, calling prepp
        # Finally we return a dictionary of results, including the costs, latencies, and the acquisition_steps
        raw_results_dict = ...
        return raw_results_dict

    def _adjust_for_local_events(self, results: Dict, n: int, s_current: SolutionCandidate) -> Dict:
        # here we adjust the costs based on events already present at the target node 'n'.
        # Using the s_current.event_stack to determine which events are locally available
        # This should be the logic from handle_intersection and _adjust_acquisition_steps
        # Should return the same dictionary as was given in, but the costs, latency and acquisition steps adjusted
        # Keep in mind that costs are only the local costs and not the global costs
        final_results = results.copy()
        return final_results

    def _format_results(self, final_results: Dict) -> Dict:
        # Here we convert the final calculated values into clean standardized list dictionaries, one for each valid strategy
        # Format example for all push:
        formatted_results = []
        push_results = {
            "strategy": "all_push",
            "individual_cost": final_results["all_push_cost"],
            "transmission_latency": final_results["all_push_transmission_latency"],
            "processing_latency": final_results["all_push_processing_latency"],
            "acquisition_steps": final_results["all_push_acquisition_steps"],
        }

        formatted_results.append(push_results)

        # PrePP is also able to say that there is no real push pull strategy that is beneficial, in this case the all push
        # and push pull strategy are the same so we need to check if they are different to not return two same states.
        if not_the_same_strat:
            pull_results = {
                "strategy": "push_pull",
                "individual_cost": final_results["push_pull_cost"],
                "transmission_latency": final_results["push_pull_transmission_latency"],
                "processing_latency": final_results["push_pull_processing_latency"],
                "acquisition_steps": final_results["push_pull_acquisition_steps"],
            }

        formatted_results.append(pull_results)
        return formatted_results