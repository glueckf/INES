"""Regression test for simulation_environment.compute_all_push().

Used to iterate once per producer *node* of an event type but multiply the
event type's already-summed total rate (h_rates_data) by the distance to
whichever single producer happened to be closest -- so an event type
produced at k nodes had its true cost inflated by a factor of k (e.g.
medium/seq_abcd: reported 33720.0 where the correct sum-of-producers cost
is 11244.0). This is the same bug, independently reimplemented, as the one
fixed in src/inev/placement_aug.py (see src/inev/tests/test_placement_aug.py)
-- found via a user report that manually placing every operator of a query
at the cloud with all-push chosen gave a different total than the "All-Push"
leaderboard baseline for the identical placement+strategy.

Fixed by summing each producer's own rate times its own distance directly
from h_local_rate_lookup, exactly as
kraken.components.cost_calculator.CostCalculator._compute_all_push_costs
already does.
"""
import contextlib
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "src")
for path in (REPO, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import simulation_environment as se  # noqa: E402
from core.query_workload import number_children  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "demo", "export"))
import scenarios_def  # noqa: E402


class TestComputeAllPush(unittest.TestCase):
    """Same real (12-node, deterministic) scenario used by
    src/inev/tests/test_placement_aug.py and this session's export
    pipeline -- real topology and rates, no synthetic fixture."""

    @classmethod
    def setUpClass(cls):
        spec = next(s for s in scenarios_def.SCENARIOS if s["id"] == "seq_abcd")
        q = number_children(spec["build"]())
        se.generate_hardcoded_workload = lambda: [q]
        cfg = se.SimulationConfig.create_deterministic(
            network_size=12, num_event_types=6, xi=0.0, cost_weight=0.5,
            latency_threshold=None, output_dataset_name="test_simulation_environment",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            cls.sim = se.Simulation(cfg)
            cls.sim.run()

    def test_matches_sum_of_producers(self):
        sim = self.sim
        # Independent cross-check: sum each producer's own rate times its
        # own distance to the cloud (node 0), for every leaf event type
        # in the workload -- the model compute_all_push is supposed to
        # implement.
        dest_distances = sim.allPairs[0]
        eventtypes = set()
        for q in sim.query_workload:
            eventtypes.update(q.leafs())
        expected = sum(
            rate * dest_distances[node_id]
            for et in eventtypes
            for node_id, rate in sim.h_local_rate_lookup.get(et, {}).items()
        )
        self.assertAlmostEqual(sim.all_push_results["cost"], expected)
        # Pin the concrete number too, not just internal self-consistency.
        self.assertAlmostEqual(sim.all_push_results["cost"], 11244.0)


if __name__ == "__main__":
    unittest.main()
