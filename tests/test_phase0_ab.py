import random
import unittest

from experiments.phase0_ab import counterfactual_utilities, make_episode, run_episode


class Phase0Tests(unittest.TestCase):
    def test_oracle_preserves_delayed_target(self):
        episode = make_episode(0, 48, random.Random(7), "delayed_query")
        metrics = run_episode(episode, capacity=4, policy="oracle", seed=1)
        self.assertEqual(metrics.success, 1.0)
        self.assertEqual(metrics.required_recall, 1.0)

    def test_stale_state_has_zero_future_utility(self):
        episode = make_episode(1, 48, random.Random(11), "state_overwrite")
        updates = [r for r in episode.records if r.key == episode.target_key]
        decision_pos = updates[-1].position
        utility = counterfactual_utilities(episode, updates, decision_pos)
        for old in updates[:-1]:
            self.assertEqual(utility[old.uid], 0.0)
        self.assertGreater(utility[updates[-1].uid], 0.0)

    def test_oracle_handles_relation_synergy(self):
        episode = make_episode(2, 64, random.Random(13), "relation_chain")
        metrics = run_episode(episode, capacity=4, policy="oracle", seed=2)
        self.assertEqual(metrics.success, 1.0)


if __name__ == "__main__":
    unittest.main()
