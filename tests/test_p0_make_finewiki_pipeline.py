import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p0_pipeline import validate  # noqa: E402
from p0_verify_finewiki_stage import STAGES  # noqa: E402


class FineWikiPipelineConfigTest(unittest.TestCase):
    def test_all_stages_are_cpu_only_and_independently_checked(self):
        with tempfile.TemporaryDirectory() as d:
            config_path = Path(d) / "pipeline.json"
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/p0_make_finewiki_pipeline.py"),
                "--root", "/remote/finewiki", "--reference-root", "/remote/wave1",
                "--tokenizer", "/remote/tokenizer.json", "--python", "/remote/python",
                "--output", str(config_path),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(config_path.read_text())
            validate(config)
            self.assertEqual({s["name"] for s in config["stages"]}, set(STAGES))
            self.assertEqual(config["paid_stages"], 0)
            self.assertEqual(config["gpu_stages"], 0)
            for stage in config["stages"]:
                self.assertEqual(stage["kind"], "command")
                self.assertEqual(stage["host"], "CPU_NODE")
                self.assertEqual(len(stage["checks"]), 1)
                self.assertIn("p0_verify_finewiki_stage.py", stage["checks"][0]["command"])
                self.assertNotIn("DEEPSEEK_API_KEY", stage["command"])


if __name__ == "__main__":
    unittest.main()
