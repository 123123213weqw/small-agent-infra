import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p0_tokenizer_corpus", ROOT / "scripts" / "p0_tokenizer_corpus.py")
CORPUS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CORPUS)


class P0TokenizerCorpusTests(unittest.TestCase):
    def write_source(self, root: Path, source_id: str, domain: str) -> None:
        directory = root / source_id
        directory.mkdir(parents=True)
        rows = []
        for index in range(6):
            text = f"{domain} document {index} abcdefghijklmnopqrstuvwxyz"
            rows.append(
                {
                    "document_id": CORPUS.stable_rank(index, source_id),
                    "source_id": source_id,
                    "source_revision": "rev",
                    "source_locator": f"row={index}",
                    "family_id": f"{source_id}-family-{index}",
                    "domain": domain,
                    "language": "zh" if domain.endswith("zh") else "en",
                    "text": text,
                    "normalized_sha256": CORPUS.stable_rank(9, text),
                    "license_status": "pending",
                    "split": "train" if index < 4 else "validation",
                    "rule_keep": True,
                }
            )
        shard = directory / "part-00000.parquet"
        pq.write_table(pa.Table.from_pylist(rows), shard)
        manifest = {
            "stage": "normalize",
            "source_id": source_id,
            "shards": [{
                "file": shard.name,
                "bytes": shard.stat().st_size,
                "sha256": CORPUS.sha256_file(shard),
                "rows": len(rows),
            }],
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_build_balanced_corpus_and_holdout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalized = root / "normalized"
            self.write_source(normalized, "zh-source", "general_zh")
            self.write_source(normalized, "en-source", "general_en")
            config = {
                "seed": 7,
                "corpus": {
                    "target_characters": 100,
                    "domain_character_targets": {"general_zh": 50, "general_en": 50},
                    "max_documents_per_family": 4,
                },
                "shared": {"special_tokens": ["<eos>"]},
                "evaluation": {"heldout_documents_per_domain": 1},
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "corpus"
            args = types.SimpleNamespace(
                config=config_path,
                normalized_root=normalized,
                output=output,
                oversample=2.0,
            )
            result = CORPUS.build(args)
            self.assertGreaterEqual(result["counts"]["train_characters"], 100)
            self.assertEqual(result["counts"]["heldout_documents"], 2)
            self.assertEqual(len(result["train_shards"]), 2)
            self.assertFalse(result["license_gate"]["training_eligible"])


if __name__ == "__main__":
    unittest.main()
