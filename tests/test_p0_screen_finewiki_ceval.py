import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_screen_finewiki_ceval import normalize  # noqa: E402


class CevScreenTest(unittest.TestCase):
    def test_normalization_is_punctuation_insensitive_and_one_line(self):
        self.assertEqual(normalize("税法（A） \n原则：１２3？"), "税法a原则123")
        self.assertEqual(normalize("ＡＢＣ"), "abc")

    def test_grep_emits_only_line_and_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "patterns.txt"
            phrase = normalize("下列关于税法基本原则的表述中，不正确的是____。")
            path.write_text(phrase + "\n")
            text = "无关资料\n" + normalize("下列关于税法基本原则的表述中不正确的是。") + "答D\n"
            result = subprocess.run(["grep", "-aFon", "-f", str(path)], input=text,
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), "2:" + phrase)


if __name__ == "__main__":
    unittest.main()
