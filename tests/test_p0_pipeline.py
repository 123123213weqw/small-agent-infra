import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "p0_pipeline.py"


class PipelineTests(unittest.TestCase):
    def invoke(self, action, config, state, *extra):
        return subprocess.run([sys.executable, str(SCRIPT), action, "--config", str(config),
                               "--state", str(state), *extra], capture_output=True, text=True)

    def test_run_resume_and_mutation_stop(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source, output = root / "source", root / "output"
            source.write_text("first")
            config, state = root / "config.json", root / "state.json"
            config.write_text(json.dumps({"version": 1, "stages": [{
                "name": "copy_text", "kind": "command", "host": "local",
                "inputs": [{"host": "local", "path": str(source)}],
                "outputs": [{"host": "local", "path": str(output)}],
                "command": f"cp {source} {output}",
            }]}))
            self.assertEqual(self.invoke("run", config, state).returncode, 0)
            self.assertIn("SKIP copy_text", self.invoke("run", config, state).stdout)
            output.write_text("tampered")
            self.assertNotEqual(self.invoke("run", config, state).returncode, 0)

    def test_paid_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            output, config, state = root / "output", root / "config.json", root / "state.json"
            config.write_text(json.dumps({"version": 1, "stages": [{
                "name": "judge", "kind": "paid", "host": "local", "budget_usd": 5,
                "expected_key_sha256": "a" * 64,
                "command": f"printf '%s' '{{budget_usd}} {{expected_key_sha256}}' > {output}",
                "outputs": [{"host": "local", "path": str(output)}],
                "checks": [{"host": "local", "command": f"test -s {output}"}],
            }]}))
            self.assertNotEqual(self.invoke("run", config, state).returncode, 0)
            self.assertFalse(output.exists())
            self.assertNotEqual(self.invoke("run", config, state, "--allow-paid", "--max-paid-usd", "4").returncode, 0)
            self.assertEqual(self.invoke("run", config, state, "--allow-paid", "--max-paid-usd", "5").returncode, 0)

    def test_untracked_output_requires_adoption_check(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            output, config, state = root / "output", root / "config.json", root / "state.json"
            output.write_text("old")
            config.write_text(json.dumps({"version": 1, "stages": [{
                "name": "stage", "kind": "command", "host": "local", "command": "false",
                "outputs": [{"host": "local", "path": str(output)}],
                "checks": [{"host": "local", "command": f"test -s {output}"}],
            }]}))
            self.assertNotEqual(self.invoke("run", config, state).returncode, 0)
            self.assertEqual(self.invoke("adopt", config, state, "--stage", "stage").returncode, 0)
            self.assertIn("SKIP stage", self.invoke("run", config, state).stdout)

    def test_adopt_all_verifies_existing_stages_without_running_commands(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            first, second = root / "first.txt", root / "second.txt"
            first.write_text("one")
            second.write_text("two")
            config, state = root / "config.json", root / "state.json"
            config.write_text(json.dumps({"version": 1, "stages": [
                {"name": "first", "kind": "command", "host": "local", "command": "false",
                 "outputs": [{"host": "local", "path": str(first)}],
                 "checks": [{"host": "local", "command": f"test -s {first}"}]},
                {"name": "second", "kind": "command", "host": "local", "command": "false",
                 "inputs": [{"host": "local", "path": str(first)}],
                 "outputs": [{"host": "local", "path": str(second)}],
                 "checks": [{"host": "local", "command": f"test -s {second}"}]},
            ]}))
            result = self.invoke("adopt-all", config, state)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(json.loads(state.read_text())["stages"]), {"first", "second"})
            self.assertEqual(self.invoke("run", config, state).returncode, 0)


if __name__ == "__main__":
    unittest.main()
