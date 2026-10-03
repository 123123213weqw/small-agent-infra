import json
import random
import tempfile
import unittest
from pathlib import Path

from experiments.phase0_ab import make_episode
from experiments.phase0_b21_data import (
    COMPOSITION_TEMPLATES,
    FORBIDDEN_INPUT_FIELDS,
    PARAPHRASE_TEMPLATES,
    RELATION_AUG_EDGE_TEMPLATES,
    ROBUST_TRAIN_EDGE_TEMPLATES,
    ROBUST_VALIDATION_EDGE_TEMPLATES,
    SplitSpec,
    build_dataset,
    collect_episode_groups,
    composition_unseen_characters,
    group_model_input,
    validate_group,
)


class Phase0B21DataTests(unittest.TestCase):
    def test_robust_relation_grammar_has_disjoint_validation_and_frozen_tests(self):
        frozen = {COMPOSITION_TEMPLATES["edge"], PARAPHRASE_TEMPLATES["edge"][1]}
        self.assertTrue(set(ROBUST_TRAIN_EDGE_TEMPLATES).isdisjoint(frozen))
        self.assertTrue(set(ROBUST_VALIDATION_EDGE_TEMPLATES).isdisjoint(frozen))
        self.assertTrue(
            set(ROBUST_TRAIN_EDGE_TEMPLATES).isdisjoint(ROBUST_VALIDATION_EDGE_TEMPLATES)
        )

    def test_robust_relation_keeps_frozen_tests_and_volume(self):
        specs = (
            SplitSpec("train", 12, (32,), ("A", "B", "C")),
            SplitSpec("validation", 12, (32,), ("A", "B", "C")),
            SplitSpec("test_composition", 12, (32,), ("D",)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = build_dataset(root / "baseline", "audit", seed=0, specs=specs)
            robust = build_dataset(
                root / "robust", "audit_relation_robust", seed=0, specs=specs
            )
            self.assertEqual(
                baseline["splits"]["train"]["decisions"],
                robust["splits"]["train"]["decisions"],
            )
            self.assertEqual(
                baseline["splits"]["test_composition"]["sha256"],
                robust["splits"]["test_composition"]["sha256"],
            )
            self.assertTrue(
                any(
                    key.startswith("RT")
                    for key in robust["splits"]["train"]["template_counts"]
                )
            )
            self.assertTrue(
                any(
                    key.startswith("RV")
                    for key in robust["splits"]["validation"]["template_counts"]
                )
            )

    def test_relation_augmentation_keeps_frozen_tests_and_volume(self):
        specs = (
            SplitSpec("train", 12, (32,), ("A", "B", "C")),
            SplitSpec("validation", 12, (32,), ("A", "B", "C")),
            SplitSpec("test_composition", 12, (32,), ("D",)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = build_dataset(root / "baseline", "audit", seed=0, specs=specs)
            augmented = build_dataset(
                root / "augmented", "audit_relation_aug", seed=0, specs=specs
            )
            self.assertEqual(
                baseline["splits"]["train"]["decisions"],
                augmented["splits"]["train"]["decisions"],
            )
            self.assertEqual(
                baseline["splits"]["test_composition"]["sha256"],
                augmented["splits"]["test_composition"]["sha256"],
            )
            relation_families = {
                key
                for key in augmented["splits"]["train"]["template_counts"]
                if key.startswith("R")
            }
            self.assertTrue(relation_families)

    def test_relation_augmentation_does_not_copy_frozen_edge_templates(self):
        frozen = {COMPOSITION_TEMPLATES["edge"], PARAPHRASE_TEMPLATES["edge"][1]}
        self.assertTrue(frozen.isdisjoint(RELATION_AUG_EDGE_TEMPLATES))

    def test_composition_family_uses_seen_character_vocabulary(self):
        self.assertEqual(composition_unseen_characters(), {})

    def test_group_is_complete_causal_decision(self):
        episode = make_episode(101, 32, random.Random(12), "state_overwrite")
        groups, _ = collect_episode_groups(
            episode,
            split="train",
            capacity=4,
            hard_negative_count=4,
            template_family="A",
            behavior_policy="fifo",
            seed=7,
        )
        self.assertEqual(len(groups), 12)
        for group in groups:
            self.assertEqual(validate_group(group), [])
            self.assertEqual(len(group["records"]), group["capacity"] + 1)
            self.assertEqual(len(group["utilities"]), len(group["records"]))
            self.assertTrue(group["records"][group["candidate_index"]]["is_candidate"])
            self.assertTrue(
                all(record["event_index"] <= group["decision_position"] for record in group["records"])
            )
            self.assertTrue(
                all(
                    record["event_index"] < group["decision_position"]
                    for record in group["recent_context"]
                )
            )

    def test_model_input_excludes_labels_and_structure_truth(self):
        episode = make_episode(102, 32, random.Random(13), "completed_intermediate")
        groups, _ = collect_episode_groups(
            episode,
            split="validation",
            capacity=8,
            hard_negative_count=8,
            template_family="C",
            behavior_policy="oracle",
            seed=8,
        )
        serialized = json.dumps(group_model_input(groups[0]), ensure_ascii=False)
        for field in FORBIDDEN_INPUT_FIELDS:
            self.assertNotIn(f'"{field}"', serialized)
        self.assertNotIn("decision_position", serialized)
        self.assertNotIn("behavior_policy", serialized)

    def test_generation_is_deterministic(self):
        episode = make_episode(103, 48, random.Random(14), "relation_chain")
        kwargs = dict(
            split="test_composition",
            capacity=12,
            hard_negative_count=14,
            template_family="D",
            behavior_policy="random",
            seed=9,
        )
        first, first_entities = collect_episode_groups(episode, **kwargs)
        second, second_entities = collect_episode_groups(episode, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first_entities, second_entities)

    def test_split_files_and_entities_are_isolated(self):
        specs = (
            SplitSpec("train", 6, (32,), ("A", "B", "C")),
            SplitSpec("test_composition", 6, (32,), ("D",)),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            manifest = build_dataset(output, "audit", seed=0, specs=specs)
            self.assertEqual(set(manifest["splits"]), {"train", "test_composition"})
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "AUDIT.md").is_file())
            train = (output / "train.jsonl").read_text(encoding="utf-8")
            test = (output / "test_composition.jsonl").read_text(encoding="utf-8")
            self.assertIn("entity_e100000000_", train)
            self.assertNotIn("entity_e200000000_", train)
            self.assertIn("entity_e200000000_", test)
            self.assertNotIn('"template_family":"D"', train)


if __name__ == "__main__":
    unittest.main()
