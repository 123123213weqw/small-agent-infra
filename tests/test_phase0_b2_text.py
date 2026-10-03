import random
import tempfile
import unittest
from pathlib import Path

from experiments.phase0_ab import make_episode
from experiments.experiment_visualization import write_eviction_artifacts
from experiments.phase0_b2_text import (
    TextRenderer,
    encode_bytes,
    harden_episode,
    lexical_scores,
    run_scored_episode,
)


class Phase0B2Tests(unittest.TestCase):
    def test_model_input_contains_no_literal_structure_flags(self):
        episode = make_episode(10, 32, random.Random(4), "completed_intermediate")
        renderer = TextRenderer("train")
        candidate = episode.records[5]
        text = renderer.model_input(episode, candidate, episode.records[:6], 5)
        self.assertNotIn("consumed=", text)
        self.assertNotIn("unresolved=", text)
        self.assertNotIn("goal_match=", text)

    def test_paraphrase_templates_are_disjoint(self):
        episode = make_episode(11, 32, random.Random(5), "state_overwrite")
        record = episode.records[4]
        self.assertNotEqual(TextRenderer("train").record(record), TextRenderer("paraphrase").record(record))

    def test_byte_encoding_is_fixed_length(self):
        ids, mask = encode_bytes("目标：测试。", 32)
        self.assertEqual(tuple(ids.shape), (32,))
        self.assertEqual(tuple(mask.shape), (32,))
        self.assertGreater(int(mask.sum()), 1)

    def test_hard_negatives_do_not_replace_required_records(self):
        episode = make_episode(12, 48, random.Random(8), "unresolved_subgoal")
        original = {r.uid: r for r in episode.records}
        hardened = harden_episode(episode, 99)
        for uid in episode.required_ids:
            self.assertEqual(original[uid], next(r for r in hardened.records if r.uid == uid))
        overlaps = [r for r in hardened.records if r.uid not in episode.required_ids and r.key in episode.goal_tokens]
        self.assertGreater(len(overlaps), 0)

    def test_eviction_trace_contains_only_observed_competitors(self):
        episode = make_episode(13, 32, random.Random(9), "state_overwrite")
        renderer = TextRenderer("train")
        traces = []
        run_scored_episode(
            episode,
            4,
            lambda ep, records, pos: lexical_scores(renderer, ep, records, pos),
            renderer=renderer,
            split="test",
            policy="lexical",
            trace=traces,
        )
        self.assertGreater(len(traces), 0)
        for decision in traces:
            self.assertEqual(len(decision["records"]), 5)
            self.assertEqual(sum(bool(r["evicted"]) for r in decision["records"]), 1)
            self.assertTrue(
                all(r["position"] <= decision["decision_position"] for r in decision["records"])
            )

    def test_eviction_viewer_and_jsonl_are_written(self):
        trace = {
            "split": "id",
            "policy": "text",
            "episode": 1,
            "task": "state_overwrite",
            "capacity": 1,
            "decision": 1,
            "decision_position": 1,
            "goal": "测试目标",
            "candidate_uid": "b",
            "evicted_uid": "a",
            "oracle_evictions": ["a"],
            "regret": 0.0,
            "episode_success": True,
            "records": [
                {
                    "uid": "a",
                    "text": "旧记录",
                    "position": 0,
                    "age": 1,
                    "is_candidate": False,
                    "utility": 0.0,
                    "score": 0.1,
                    "evicted": True,
                    "oracle": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_eviction_artifacts(output, [trace])
            self.assertIn("测试目标", (output / "eviction_trace.jsonl").read_text(encoding="utf-8"))
            viewer = (output / "eviction_viewer.html").read_text(encoding="utf-8")
            self.assertIn("记忆淘汰查看器", viewer)
            self.assertIn("旧记录", viewer)


if __name__ == "__main__":
    unittest.main()
