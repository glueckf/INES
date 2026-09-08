"""Regression tests for the placement-cost model in placement_aug.py.

Both new_compute_central_costs and compute_single_sink_placement's
primitive-event case used to compute a primitive event type's transmission
cost as (that event type's already-summed total rate across every
producer) x (distance from the single closest producer to the
destination), and then add that same figure again once per producer
instance -- so an event type produced at k nodes had its true cost
inflated by a factor of k. The correct model (confirmed against the user
2026-09-09: "it is tuple count * hop per source") is the tuple rate at
each source times that source's own hop distance to the destination,
summed over every source -- the same model
CostCalculator._compute_all_push_costs (kraken/components/
cost_calculator.py) and simulation_environment.compute_all_push() already
use.

Found via a real discrepancy: the "INEv" baseline exported for the demo
reported costs 3-6x too high across every scenario, including one case
(large/seq_abc, a single-operator query) where INEv's placement was
byte-identical to the "All-Push" baseline's -- same node, same push
strategy -- yet its reported cost was 5.6x higher, which is only possible
if the cost function itself was wrong. Both instances of the bug are
independent reimplementations of the same flawed pattern (see also the
now-fixed simulation_environment.compute_all_push, which had the identical
issue).

Verified (2026-09-09) across all 8 shipped demo scenarios that this fix
does not change which node INEv places anything on -- only the reported
cost. That is a property of the specific scenarios tested, not a general
guarantee; these tests exist so a future change to this cost model can be
checked against known-correct numbers instead of re-deriving them by hand.
"""
import contextlib
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "src")
for path in (REPO, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

if os.environ.get("PYTHONHASHSEED") != "0":
    # PrePP seeds random.seed(42 + hash(query)); match export_scenario.py /
    # score_one.py so a run of this file alone is still reproducible.
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import simulation_environment as se  # noqa: E402
from core.query_workload import number_children  # noqa: E402
from inev.placement_aug import (  # noqa: E402
    compute_single_sink_placement,
    new_compute_central_costs,
)

sys.path.insert(0, os.path.join(REPO, "demo", "export"))
import scenarios_def  # noqa: E402


def _build_seq_abcd_medium_config():
    spec = next(s for s in scenarios_def.SCENARIOS if s["id"] == "seq_abcd")
    q = number_children(spec["build"]())
    se.generate_hardcoded_workload = lambda: [q]
    return se.SimulationConfig.create_deterministic(
        network_size=12, num_event_types=6, xi=0.0, cost_weight=0.5,
        latency_threshold=None, output_dataset_name="test_placement_aug",
    )


def _run_seq_abcd_medium():
    """The same small (12-node, deterministic) scenario used throughout
    the demo export pipeline and this session's own verification -- real
    topology, real rates, no synthetic fixture, so these tests exercise
    exactly the code path the shipped demo data went through."""
    with contextlib.redirect_stdout(io.StringIO()):
        sim = se.Simulation(_build_seq_abcd_medium_config())
        sim.run()
    return sim


def _fresh_seq_abcd_medium():
    """Simulation.__init__ alone (no .run()) already populates every h_*
    attribute compute_single_sink_placement needs -- deliberately used
    instead of _run_seq_abcd_medium() here: compute_single_sink_placement
    mutates EventNodes in place (via set_event_nodes) as each projection in
    the processing order gets placed, so calling it a second time against
    an already-.run() simulation would see the *final*, fully-mutated
    network state rather than what the real search actually saw when it
    first evaluated this projection."""
    with contextlib.redirect_stdout(io.StringIO()):
        sim = se.Simulation(_build_seq_abcd_medium_config())
    return sim


class TestNewComputeCentralCosts(unittest.TestCase):
    """new_compute_central_costs always evaluates at destination=0, the
    same node compute_all_push() prices -- for the *same* workload, both
    functions are computing the identical quantity (every producer of
    every leaf event type sends its own stream to node 0), so they must
    agree exactly once both use the correct per-producer model."""

    @classmethod
    def setUpClass(cls):
        cls.sim = _run_seq_abcd_medium()

    def test_matches_verified_all_push_cost(self):
        sim = self.sim
        costs, node, _longest_path, _routing = new_compute_central_costs(
            sim.query_workload, sim.h_IndexEventNodes, sim.allPairs,
            sim.h_rates_data, sim.h_eventNodes, sim.graph, sim.h_local_rate_lookup,
        )
        self.assertEqual(node, 0)
        # compute_all_push() is independently verified (see
        # simulation_environment.py's own docstring/history) against a
        # hand-computed sum-of-producers cost for this exact scenario.
        self.assertAlmostEqual(costs, sim.all_push_results["cost"])
        # Pin the concrete number too, not just cross-function agreement --
        # two functions sharing the *same* bug would also "agree".
        self.assertAlmostEqual(costs, 11244.0)


class TestComputeSingleSinkPlacementPrimitiveCost(unittest.TestCase):
    """SEQ(A, B) depends only on raw primitives (no sub-query, no filter),
    so its whole cost goes through the one branch this fix touched --
    letting this test pin an exact, hand-verifiable number instead of the
    fuzzier "did the total INEv cost go down" check."""

    @classmethod
    def setUpClass(cls):
        cls.sim = _fresh_seq_abcd_medium()
        cls.seq_ab = next(
            p for p in cls.sim.h_mycombi.keys() if str(p) == "SEQ(A, B)"
        )

    def test_seq_ab_placement_cost(self):
        sim = self.sim
        combination = sim.h_mycombi[self.seq_ab]
        self.assertEqual(sorted(str(c) for c in combination), ["A", "B"])

        with contextlib.redirect_stdout(io.StringIO()):
            cost, node, _longest_path, _proj, _new_instances, _filters, _proc_lat = (
                compute_single_sink_placement(
                    self.seq_ab, combination, 0, sim.h_projFilterDict,
                    sim.h_eventNodes, sim.h_IndexEventNodes, sim.h_network_data,
                    sim.allPairs, sim.h_mycombi, sim.h_rates_data,
                    sim.single_selectivity, sim.h_projrates, sim.graph,
                    sim.network, sim,
                )
            )

        # Node 4 wins under the fixed formula (same winner as before the
        # fix, for this scenario -- see class/module docstring), with a
        # cost of sum(rate * distance) over A's 3 real producers and B's 2,
        # cross-checked directly against sim.h_local_rate_lookup by hand.
        self.assertEqual(node, 4)
        self.assertAlmostEqual(cost, 3004.0)


if __name__ == "__main__":
    unittest.main()
