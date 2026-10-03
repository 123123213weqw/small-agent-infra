import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p0_make_token_mixture.py"


class TokenMixtureTests(unittest.TestCase):
    def test_train_only_proportions(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "source"
            source.mkdir()
            (source / "manifest.json").write_text(json.dumps({"stage": "accepted_real",
                "license_gate": {"training_eligible": True}}))
            rows = [
                {"split": "train", "domain": "code_python", "own_text_tokens": 30},
                {"split": "train", "domain": "math_en", "own_text_tokens": 70},
                {"split": "validation", "domain": "math_en", "own_text_tokens": 900},
            ]
            (source / "combined_index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
            output = root / "mix.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--source", str(source),
                                     "--output", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text()), {"code_python": 0.3, "math_en": 0.7})
            self.assertNotEqual(subprocess.run([sys.executable, str(SCRIPT), "--source", str(source),
                                                "--output", str(output)], capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
