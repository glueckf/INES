"""Cost calculation pipeline for placement decisions."""

import io
import hashlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.kraken2_0.state import SolutionCandidate
from src.prepp import generate_prePP


class CostCalculator:
    """
    Modular component for cost and latency calculation pipeline.

    Encapsulates the logic for calling the PrePP function, adjusting
    for local events, and adding final sink costs. This is a completely
    self-contained implementation that uses only the SolutionCandidate
    state and the external PrePP library.
    """

    def __init__(self, context: Dict[str, Any]):
        """Initialize calculator with static problem data.

        Args:
            context: Dictionary containing all static problem parameters
        """
        self.params = context
        self._prepp_cache: Dict[str, Any] = {}

    def calculate(
        self, p: Any, n: int, s_current: SolutionCandidate
    ) -> List[Dict[str, Any]]:
        """Orchestrate the full cost calculation pipeline.

        Args:
            p: Projection being placed
            n: Node ID where placement is considered
            s_current: Current solution state

        Returns:
            List of dictionaries with strategy-specific costs and metrics
        """
        # Get raw costs for all strategies
        raw_strategies = self._get_raw_costs_for_strategies(p, n, s_current)

        # Apply adjustments to each strategy
        final_strategies = []
        for strategy in raw_strategies:
            adjusted = self._adjust_for_local_events(strategy, p, n, s_current)
            final = self._add_sink_costs(adjusted, p, n)
            final_strategies.append(final)

        return final_strategies

    def _get_raw_costs_for_strategies(
        self, p: Any, n: int, s_current: SolutionCandidate
    ) -> List[Dict[str, Any]]:
        """Get raw costs for all valid strategies.

        Args:
            p: Projection being placed
            n: Node ID where placement is considered
            s_current: Current solution state

        Returns:
            List of strategy dictionaries (all-push and optionally push-pull)
        """
        # Step 1: Compute all-push strategy
        all_push_result = self._compute_all_push_costs(p, n, s_current)

        # Step 2: Compute push-pull strategy using PrePP
        is_deterministic = self.params["simulation_mode"].value == "deterministic"

        METHOD = "ppmuse"
        ALGORITHM = "e"
        SAMPLES = 0
        TOP_K = 0
        RUNS = 1
        PLAN_PRINT = True

        all_pairs = self.params.get("all_pairs")

        # Create input buffer for PrePP
        input_buffer = self._build_prepp_input_buffer(p, n, s_current)
        if input_buffer is None:
            return [all_push_result]

        # Run PrePP for push-pull optimization
        prepp_output = generate_prePP(
            input_buffer=input_buffer,
            method=METHOD,
            algorithm=ALGORITHM,
            samples=SAMPLES,
            top_k=TOP_K,
            runs=RUNS,
            plan_print=PLAN_PRINT,
            allPairs=all_pairs,
            is_deterministic=is_deterministic,
        )

        # Process PrePP results to extract push-pull strategy
        if not prepp_output or len(prepp_output) < 6:
            return [all_push_result]

        push_pull_result = self._process_prepp_output(
            prepp_output, p, n, all_push_result["individual_cost"]
        )

        # Step 3: Compare and return distinct strategies
        if push_pull_result is None:
            return [all_push_result]

        # Check if strategies are identical
        strategies_identical = (
            push_pull_result["individual_cost"] == all_push_result["individual_cost"]
            and push_pull_result["transmission_latency"]
            == all_push_result["transmission_latency"]
            and push_pull_result["processing_latency"]
            == all_push_result["processing_latency"]
        )

        if strategies_identical:
            return [all_push_result]

        return [all_push_result, push_pull_result]

    def _adjust_for_local_events(
        self, strategy_result: Dict, p: Any, n: int, s_current: SolutionCandidate
    ) -> Dict:
        """Adjust costs for events already available at node.

        Args:
            strategy_result: Single strategy dictionary to adjust
            p: Projection being placed
            n: Node ID where placement is considered
            s_current: Current solution state

        Returns:
            Adjusted strategy dictionary
        """
        # Get events available at node from state
        if n not in s_current.event_stack:
            return strategy_result

        available_events = set(s_current.event_stack[n].keys())

        # Extract primitive events needed for projection
        needed_events = self._extract_needed_events(p)

        if not (available_events & needed_events):
            return strategy_result

        events_already_available = available_events & needed_events

        # Calculate adjustment based on strategy type
        strategy_name = strategy_result["strategy"]
        acquisition_steps = strategy_result["acquisition_steps"]

        if strategy_name == "all_push":
            cost_adj, lat_adj = self._compute_all_push_adjustment(
                acquisition_steps, events_already_available
            )
        else:  # push_pull
            cost_adj, lat_adj = self._compute_acquisition_adjustment(
                acquisition_steps, events_already_available, n, s_current
            )

        # Apply adjustments
        strategy_result["individual_cost"] = max(
            0.0, strategy_result["individual_cost"] - cost_adj
        )
        strategy_result["transmission_latency"] = max(
            0.0, strategy_result["transmission_latency"] - lat_adj
        )

        return strategy_result

    def _add_sink_costs(self, strategy_result: Dict, p: Any, n: int) -> Dict:
        """Add transmission costs to sink nodes if needed.

        Args:
            strategy_result: Single strategy dictionary to update
            p: Projection being placed
            n: Node ID where placement is considered

        Returns:
            Strategy dictionary with sink costs added
        """
        # Check if projection needs to be sent to sinks
        if p not in self.params["query_workload"] or n in self.params["sink_nodes"]:
            return strategy_result

        proj_rates = self.params["projection_rates_selectivity"]
        output_rate = proj_rates.get(p, (0, 1))[1]

        # Calculate distances to sinks
        sink_nodes = self.params["sink_nodes"]
        distances = [
            self.params["pairwise_distance_matrix"][n][sink] for sink in sink_nodes
        ]

        if distances:
            total_distance = sum(distances)
            transmission_cost = total_distance * output_rate
            max_distance = max(distances)

            strategy_result["individual_cost"] += transmission_cost
            strategy_result["transmission_latency"] += max_distance

        return strategy_result

    def _create_cache_key(self, p: Any, n: int) -> str:
        """Create deterministic cache key."""
        proj_str = str(p)
        sel_rate = self.params["projection_rates_selectivity"].get(p, (0.0, 0.0))[0]
        sel_items = sorted(self.params["pairwise_selectivities"].items())

        key_data = f"{n}|{proj_str}|{sel_rate}|{sel_items}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_placement_context(
        self, p: Any, s_current: SolutionCandidate
    ) -> Dict[str, Any]:
        """Gather common parameters for cost calculations.

        Args:
            p: Projection being placed
            s_current: Current solution state

        Returns:
            Dictionary of common cost calculation parameters
        """
        problem_ref = self.params.get("problem_ref")
        placed_subqueries = (
            s_current.get_placed_subqueries(p, problem_ref) if problem_ref else {}
        )

        return {
            "placed_subqueries": placed_subqueries,
            "has_placed_subqueries": bool(placed_subqueries),
            "selectivities": self.params["pairwise_selectivities"],
            "selection_rate": self.params["projection_rates_selectivity"].get(
                p, (0.0, 0.0)
            )[0],
            "dependencies": self.params["dependencies_per_projection"],
            "rates": self.params["global_event_rates"],
            "projection_rates": self.params["projection_rates_selectivity"],
            "index_event_nodes": self.params["index_event_nodes"],
            "local_rate_lookup": self.params.get("local_rate_lookup", {}),
        }

    def _build_prepp_input_buffer(
        self, p: Any, n: int, s_current: SolutionCandidate
    ) -> Optional[Any]:
        """Build input buffer for PrePP using only state information.

        Args:
            p: Projection being placed
            n: Node where placement is considered
            s_current: Current solution state

        Returns:
            Input buffer for PrePP or None if creation fails
        """
        context = self._get_placement_context(p, s_current)

        # Generate the network configuration
        lines = self._generate_network_config()

        # Add selectivities
        lines.append("selectivities")
        lines.append(str(context["selectivities"]))
        lines.append("")

        # Add queries
        lines.append("queries")
        lines.append(str(p))
        lines.append("")

        # Add muse graph with placed subqueries
        lines.append("muse graph")
        self._add_muse_graph_entries(lines, p, n, context["placed_subqueries"])

        # Create buffer
        buffer = io.StringIO()
        buffer.write("\n".join(lines))
        buffer.seek(0)

        return buffer

    def _generate_network_config(self) -> List[str]:
        """Generate network configuration section."""
        lines = ["network"]
        network = self.params["network_data_nodes"]

        for i, node in enumerate(network):
            lines.append(f"Node {i} Node {i}")

            comp_power = getattr(node, "computational_power", "inf")
            memory = getattr(node, "memory", "inf")

            # Try different possible attribute names for event rates
            event_rates = (
                getattr(node, "event_rates", None)
                or getattr(node, "eventrates", None)
                or getattr(node, "Eventrates", None)
                or getattr(node, "rates", None)
                or [0] * 6
            )

            # Try different possible attribute names for children
            children = self._extract_node_ids(
                getattr(node, "children", None)
                or getattr(node, "child", None)
                or getattr(node, "Child", None),
                network,
            )

            # Try different possible attribute names for siblings
            siblings = self._extract_node_ids(
                getattr(node, "siblings", None)
                or getattr(node, "Siblings", None)
                or getattr(node, "sibling", None),
                network,
            )

            # Build parent map
            parents = self._get_parents_for_node(i, network)

            lines.append(f"Computational Power: {comp_power}")
            lines.append(f"Memory: {memory}")
            lines.append(
                f"Eventrates: {list(event_rates) if hasattr(event_rates, '__iter__') and not isinstance(event_rates, str) else event_rates}"
            )
            lines.append(f"Parents: {parents}")
            lines.append(f"Child: {children}")
            lines.append(f"Siblings: {siblings}")
            lines.append("")

        return lines

    def _add_muse_graph_entries(
        self, lines: List[str], p: Any, n: int, placed_subqueries: Dict
    ) -> None:
        """Add muse graph entries for projection."""
        proj_str = str(p)
        sel_rate = self.params["projection_rates_selectivity"].get(p, (0.0, 0.0))[0]

        if placed_subqueries:
            # Add placed subqueries first
            for subquery, subquery_node in placed_subqueries.items():
                subq_rate = self.params["projection_rates_selectivity"].get(
                    subquery, (0.001, 0.001)
                )[0]
                primitives = self._get_primitive_events(subquery)
                combination_str = "; ".join(primitives)
                lines.append(
                    f"SELECT {subquery} FROM {combination_str} ON {{{subquery_node}}} WITH selectionRate= {subq_rate}"
                )

            # Add main query referencing virtual events
            dependencies = self.params["dependencies_per_projection"].get(p, [])
            combination_str = "; ".join(str(dep) for dep in dependencies)
        else:
            # No subqueries, use primitives
            primitives = self._get_primitive_events(p)
            combination_str = "; ".join(primitives)

        lines.append(
            f"SELECT {proj_str} FROM {combination_str} ON {{{n}}} WITH selectionRate= {sel_rate}"
        )

    def _compute_all_push_costs(
        self, p: Any, n: int, s_current: SolutionCandidate
    ) -> Dict[str, Any]:
        """Compute all-push strategy costs using state information.

        Args:
            p: Projection being placed
            n: Node where placement is considered
            s_current: Current solution state

        Returns:
            Dictionary containing all-push strategy with standardized format
        """
        context = self._get_placement_context(p, s_current)
        dependencies = context["dependencies"].get(p, [])
        distances = self.params["pairwise_distance_matrix"]

        total_cost = 0.0
        max_transmission_latency = 0.0
        pull_response = {}

        for dependency in dependencies:
            dep_str = str(dependency)

            if dependency in context["local_rate_lookup"]:
                # Primitive event with multiple sources - aggregate them
                dep_cost = 0.0
                dep_latency = 0

                for source_node, rate in context["local_rate_lookup"][
                    dependency
                ].items():
                    hops = distances[source_node][n]
                    dep_cost += hops * rate
                    dep_latency = max(dep_latency, hops)

                total_cost += dep_cost
                max_transmission_latency = max(max_transmission_latency, dep_latency)

                pull_response[dep_str] = {
                    "cost_with_selectivity": dep_cost,
                    "latency": dep_latency,
                }

            elif dependency in context["placed_subqueries"]:
                # Already placed subquery - single source
                source_node = context["placed_subqueries"][dependency]
                rate = context["projection_rates"].get(dependency, (0, 1))[1]
                hops = distances[source_node][n]
                dep_cost = hops * rate

                total_cost += dep_cost
                max_transmission_latency = max(max_transmission_latency, hops)

                pull_response[dep_str] = {
                    "cost_with_selectivity": dep_cost,
                    "latency": hops,
                }

        push_processing_latency = self.params["projection_rates_selectivity"].get(
            p, (0.0, 0.0)
        )[1]

        acquisition_steps = {
            0: {
                "detailed_cost_contribution": {"pull_response": pull_response},
                "total_step_costs": total_cost,
                "total_latency": max_transmission_latency,
            }
        }

        return {
            "strategy": "all_push",
            "individual_cost": total_cost,
            "transmission_latency": max_transmission_latency,
            "processing_latency": push_processing_latency,
            "acquisition_steps": acquisition_steps,
        }

    def _process_prepp_output(
        self, prepp_output: List, p: Any, n: int, all_push_base_cost: float
    ) -> Optional[Dict[str, Any]]:
        """Process raw PrePP output to extract push-pull strategy.

        Args:
            prepp_output: Raw PrePP results
            p: Projection
            n: Node
            all_push_base_cost: Base cost from all-push strategy for ratio calculation

        Returns:
            Push-pull strategy dictionary or None if invalid
        """
        qkey = str(p)
        steps_by_proj = prepp_output[6] if len(prepp_output) > 6 else {}

        if not steps_by_proj or qkey not in steps_by_proj:
            return None

        steps = steps_by_proj[qkey]
        total_plan_costs = sum(s.get("total_step_costs", 0) for s in steps.values())
        total_plan_latency = sum(s.get("total_latency", 0) for s in steps.values())

        # Calculate processing latency based on transmission ratio
        transmission_ratio = (
            (total_plan_costs / all_push_base_cost) if all_push_base_cost else 0.0
        )

        all_push_processing_latency = self.params["projection_rates_selectivity"].get(
            p, (0.0, 0.0)
        )[1]
        push_pull_processing_latency = transmission_ratio * all_push_processing_latency

        return {
            "strategy": "push_pull",
            "individual_cost": total_plan_costs,
            "transmission_latency": total_plan_latency,
            "processing_latency": push_pull_processing_latency,
            "acquisition_steps": steps,
        }

    def _extract_needed_events(self, p: Any) -> Set[str]:
        """Extract primitive events needed for projection.

        Args:
            p: Projection

        Returns:
            Set of primitive event names
        """
        # Use fast path if available
        if hasattr(p, "leafs") and callable(p.leafs):
            return set(p.leafs())

        # Parse from string representation
        proj_str = str(p)
        proj_str = proj_str.replace("AND", "").replace("SEQ", "")
        proj_str = proj_str.replace("(", "").replace(")", "")
        proj_str = re.sub(r"[0-9]+", "", proj_str).replace(" ", "")

        if "," in proj_str:
            return set(proj_str.split(","))
        else:
            return set(list(proj_str))

    def _compute_acquisition_adjustment(
        self,
        acquisition_steps: Dict[int, Dict[str, Any]],
        events_already_available: Set[str],
        n: int,
        s_current: SolutionCandidate,
    ) -> Tuple[float, float]:
        """Compute cost/latency adjustment for locally available events.

        Args:
            acquisition_steps: Acquisition steps from PrePP
            events_already_available: Events available at node
            n: Node ID
            s_current: Current solution state

        Returns:
            Tuple of (cost_adjustment, latency_adjustment)
        """
        total_cost_adj = 0.0
        total_lat_adj = 0.0

        for step_details in acquisition_steps.values():
            events_to_pull = step_details.get("events_to_pull", [])
            if not events_to_pull:
                continue

            all_primitives = self._parse_primitive_events(events_to_pull)
            events_we_can_skip = all_primitives & events_already_available

            if not events_we_can_skip:
                continue

            # Update metadata
            step_details["already_at_node"] = list(events_we_can_skip)
            step_details["acquired_by_query"] = self._get_query_sources(
                events_we_can_skip, n, s_current
            )

            # Calculate adjustment
            if all_primitives <= events_already_available:
                # All events available
                total_cost_adj += step_details.get("total_step_costs", 0.0)
                total_lat_adj += step_details.get("total_latency", 0.0)
            else:
                # Partial availability
                total_cost_adj += self._compute_partial_cost_adjustment(
                    step_details, events_we_can_skip, all_primitives
                )

        return total_cost_adj, total_lat_adj

    def _compute_all_push_adjustment(
        self,
        push_pull_steps: Dict[int, Dict[str, Any]],
        events_already_available: Set[str],
    ) -> Tuple[float, float]:
        """Compute all-push adjustment using push-pull step 0.

        Args:
            push_pull_steps: Push-pull acquisition steps
            events_already_available: Events available at node

        Returns:
            Tuple of (cost_adjustment, latency_adjustment)
        """
        if 0 not in push_pull_steps:
            return 0.0, 0.0

        step_0 = push_pull_steps[0]
        pull_response = step_0.get("detailed_cost_contribution", {}).get(
            "pull_response", {}
        )

        if not pull_response:
            return 0.0, 0.0

        all_push_events = set(pull_response.keys())
        events_we_can_skip = all_push_events & events_already_available

        if not events_we_can_skip:
            return 0.0, 0.0

        total_cost_adj = 0.0
        max_latency = 0.0

        for event in events_we_can_skip:
            if event in pull_response:
                details = pull_response[event]
                total_cost_adj += details.get("cost_with_selectivity", 0.0)
                max_latency = max(max_latency, details.get("latency", 0))

        if all_push_events <= events_already_available:
            return total_cost_adj, max_latency
        else:
            return total_cost_adj, 0.0

    def _parse_primitive_events(self, events_to_pull: List[Any]) -> Set[str]:
        """Parse primitive events from list.

        Args:
            events_to_pull: List of events or subqueries

        Returns:
            Set of primitive event names
        """
        primitives = set()
        for event in events_to_pull:
            event_str = str(event)
            event_str = event_str.replace("AND", "").replace("SEQ", "")
            event_str = event_str.replace("(", "").replace(")", "")
            event_str = re.sub(r"[0-9]+", "", event_str).replace(" ", "")
            if "," in event_str:
                primitives.update(event_str.split(","))
            else:
                primitives.update(list(event_str))
        return primitives

    def _get_query_sources(
        self, events: Set[str], n: int, s_current: SolutionCandidate
    ) -> Dict[str, Any]:
        """Get query sources for events from state.

        Args:
            events: Set of event names
            n: Node ID
            s_current: Current solution state

        Returns:
            Dictionary mapping events to query IDs
        """
        sources = {}
        if n in s_current.event_stack:
            node_events = s_current.event_stack[n]
            for event in events:
                if event in node_events:
                    sources[event] = node_events[event].get("query_id")
        return sources

    def _compute_partial_cost_adjustment(
        self,
        step_details: Dict[str, Any],
        events_we_can_skip: Set[str],
        all_primitives: Set[str],
    ) -> float:
        """Compute partial cost adjustment.

        Args:
            step_details: Step details
            events_we_can_skip: Events available
            all_primitives: All primitive events

        Returns:
            Cost adjustment amount
        """
        pull_response = step_details.get("detailed_cost_contribution", {}).get(
            "pull_response", {}
        )

        total_adj = 0.0
        for event in events_we_can_skip:
            if event in pull_response:
                total_adj += pull_response[event].get("cost_with_selectivity", 0.0)
            else:
                fraction = len(events_we_can_skip) / len(all_primitives)
                total_adj += step_details.get("total_step_costs", 0.0) * fraction
                break

        return total_adj

    def _extract_node_ids(self, node_list: Any, network: List) -> Optional[List[int]]:
        """Extract node IDs from node objects.

        Args:
            node_list: List of node objects
            network: Full network list

        Returns:
            List of node IDs or None
        """
        if node_list is None:
            return None
        if not hasattr(node_list, "__iter__") or isinstance(node_list, str):
            return node_list

        network_lookup = {node: i for i, node in enumerate(network)}
        ids = []

        for item in node_list:
            if hasattr(item, "id"):
                ids.append(item.id)
            elif hasattr(item, "nodeID"):
                ids.append(item.nodeID)
            elif isinstance(item, int):
                ids.append(item)
            elif item in network_lookup:
                ids.append(network_lookup[item])

        return ids if ids else None

    def _get_parents_for_node(self, node_id: int, network: List) -> Optional[List[int]]:
        """Get parent nodes for a given node.

        Args:
            node_id: Node ID
            network: Full network list

        Returns:
            List of parent node IDs or None
        """
        parents = []
        for i, node in enumerate(network):
            children = self._extract_node_ids(
                getattr(node, "children", None)
                or getattr(node, "child", None)
                or getattr(node, "Child", None),
                network,
            )
            if children and node_id in children:
                parents.append(i)

        return parents if parents else None

    def _get_primitive_events(self, projection: Any) -> List[str]:
        """Get primitive events for projection.

        Args:
            projection: Projection object

        Returns:
            List of primitive event names
        """
        if hasattr(projection, "leafs") and callable(projection.leafs):
            return list(projection.leafs())

        proj_str = str(projection)
        proj_str = proj_str.replace("AND", "").replace("SEQ", "")
        proj_str = proj_str.replace("(", "").replace(")", "")
        proj_str = re.sub(r"[0-9]+", "", proj_str).replace(" ", "")

        if "," in proj_str:
            return proj_str.split(",")
        else:
            return list(proj_str)
