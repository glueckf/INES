"""Regression tests for score_one.py's push/pull choice handling.

Two bugs were found and fixed here (2026-09-08/09), both in how a player's
push/pull choice gets turned into forced_push_group:

1. Choosing to push an already-placed sub-query dependency (as opposed to a
   raw primitive) was a silent no-op: score_one.py flattened the choice to
   its leaf primitives before passing it as forced_push_group, but PrePP's
   own forced-group matching (prepp.py's `old_copy = query.
   primitive_operators`) is one-level -- a sub-query dependency appears
   there as its own name string, never expanded -- so the flattened group
   never matched anything and the optimizer's own pick was silently scored
   instead, regardless of what the player chose. Fixed by passing the
   chosen dependency's name through unflattened.
2. The explicit "push everything" (ALL_PUSH) choice used to go through the
   same forced_push_group path with every dependency flattened together --
   which hits a *different*, separately-confirmed bug where a single-group
   ("nothing left to pull") forced plan doesn't reproduce the true all-push
   cost. Fixed by bypassing forced_push_group entirely for ALL_PUSH and
   using the independently-computed all-push strategy result directly.

These run score_one.py as a real subprocess, the same way server.py
invokes it -- the RNG-isolation subprocess boundary is part of what's being
tested, not just the cost formula.
"""
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.abspath(os.path.join(HERE, ".."))
SCORE_ONE = os.path.join(EXPORT_DIR, "score_one.py")

PLACEMENT = {"SEQ(A, B)": 0, "SEQ(A, B, D)": 0, "SEQ(A, B, C, D)": 0}


def run_score_one(push_choice):
    result = subprocess.run(
        [sys.executable, SCORE_ONE, "medium", "seq_abcd",
         json.dumps(PLACEMENT), json.dumps(push_choice)],
        capture_output=True, text=True, check=True, cwd=EXPORT_DIR,
    )
    return json.loads(result.stdout)


class TestScoreOnePushChoice(unittest.TestCase):
    def test_no_choice_uses_optimizer_pick(self):
        out = run_score_one({})
        pp = out["per_placement"]["SEQ(A, B, D)"]
        self.assertEqual(pp["strategy"], "push_pull")
        self.assertAlmostEqual(pp["cost"], 1310.5563484412162)

    def test_forcing_primitive_dependency_is_honored(self):
        # 'D' is a bad choice here (pushing the higher-rate side) -- forcing
        # it should cost more than the optimizer's own pick, not silently
        # revert to it.
        out = run_score_one({"SEQ(A, B, D)": "D"})
        pp = out["per_placement"]["SEQ(A, B, D)"]
        self.assertAlmostEqual(pp["cost"], 1827.0)

    def test_forcing_sub_query_dependency_is_honored(self):
        # This is exactly the case that was silently ignored before the
        # fix: 'SEQ(A, B)' is a sub-query dependency, not a raw primitive.
        # It happens to be the cheap/optimal choice here, so this alone
        # wouldn't have caught the bug (a no-op fallback to the optimizer's
        # own pick gives the same number) -- see
        # test_forcing_primitive_dependency_is_honored above for the case
        # that actually distinguishes "forced" from "silently ignored".
        out = run_score_one({"SEQ(A, B, D)": "SEQ(A, B)"})
        pp = out["per_placement"]["SEQ(A, B, D)"]
        self.assertAlmostEqual(pp["cost"], 1310.5563484412162)

    def test_all_push_reproduces_true_all_push_cost(self):
        out = run_score_one({"SEQ(A, B, D)": "__all_push__"})
        pp = out["per_placement"]["SEQ(A, B, D)"]
        self.assertEqual(pp["strategy"], "all_push")
        self.assertAlmostEqual(pp["cost"], 1827.0)


if __name__ == "__main__":
    unittest.main()
