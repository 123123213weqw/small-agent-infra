from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from small_agent.data.sampler import StatefulDistributedSampler  # noqa: E402


class StatefulDistributedSamplerTests(unittest.TestCase):
    def test_ranks_are_equal_and_disjoint_when_dropping_tail(self) -> None:
        rank0 = StatefulDistributedSampler(11, num_replicas=2, rank=0, seed=7)
        rank1 = StatefulDistributedSampler(11, num_replicas=2, rank=1, seed=7)
        left, right = list(rank0), list(rank1)
        self.assertEqual(len(left), 5)
        self.assertEqual(len(right), 5)
        self.assertFalse(set(left) & set(right))
        self.assertEqual(len(set(left) | set(right)), 10)

    def test_resume_continues_at_exact_next_index(self) -> None:
        original = StatefulDistributedSampler(20, num_replicas=2, rank=1, seed=99)
        iterator = iter(original)
        consumed = [next(iterator) for _ in range(3)]
        state = original.state_dict()
        expected_remaining = list(iterator)

        resumed = StatefulDistributedSampler(20, num_replicas=2, rank=1, seed=99)
        resumed.load_state_dict(state)
        self.assertEqual(list(resumed), expected_remaining)
        self.assertEqual(len(consumed) + len(expected_remaining), 10)

    def test_epoch_changes_order_deterministically(self) -> None:
        one = StatefulDistributedSampler(20, seed=5)
        one.set_epoch(3)
        order = list(one)
        two = StatefulDistributedSampler(20, seed=5)
        two.set_epoch(3)
        self.assertEqual(list(two), order)
        three = StatefulDistributedSampler(20, seed=5)
        three.set_epoch(4)
        self.assertNotEqual(list(three), order)

    def test_state_configuration_drift_is_rejected(self) -> None:
        source = StatefulDistributedSampler(20, seed=5)
        next(iter(source))
        incompatible = StatefulDistributedSampler(20, seed=6)
        with self.assertRaisesRegex(ValueError, "seed"):
            incompatible.load_state_dict(source.state_dict())

    def test_padding_mode_gives_equal_rank_lengths(self) -> None:
        samplers = [
            StatefulDistributedSampler(5, num_replicas=3, rank=rank, shuffle=False, drop_last=False)
            for rank in range(3)
        ]
        values = [list(sampler) for sampler in samplers]
        self.assertEqual([len(items) for items in values], [2, 2, 2])
        self.assertEqual(values, [[0, 3], [1, 4], [2, 0]])

    def test_optimizer_step_multiple_trims_the_same_tail_across_ranks(self) -> None:
        samplers = [
            StatefulDistributedSampler(
                271,
                num_replicas=8,
                rank=rank,
                shuffle=False,
                samples_per_rank_multiple=4,
            )
            for rank in range(8)
        ]
        values = [list(sampler) for sampler in samplers]
        self.assertEqual([len(items) for items in values], [32] * 8)
        self.assertTrue(all(len(items) % 4 == 0 for items in values))
        self.assertEqual(len(set().union(*map(set, values))), 256)

    def test_optimizer_step_multiple_is_checkpoint_compatible(self) -> None:
        source = StatefulDistributedSampler(40, samples_per_rank_multiple=4)
        next(iter(source))
        incompatible = StatefulDistributedSampler(40, samples_per_rank_multiple=5)
        with self.assertRaisesRegex(ValueError, "samples_per_rank_multiple"):
            incompatible.load_state_dict(source.state_dict())


if __name__ == "__main__":
    unittest.main()
