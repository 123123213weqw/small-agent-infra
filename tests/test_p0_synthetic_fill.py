import importlib.util
from fractions import Fraction
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p0_synthetic_fill", ROOT / "scripts" / "p0_synthetic_fill.py")
SYNTHETIC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SYNTHETIC)


class P0SyntheticFillTests(unittest.TestCase):
    def test_fraction_evaluator(self):
        self.assertEqual(SYNTHETIC.evaluate_fraction("(2+3)*4/5"), Fraction(4))
        self.assertEqual(SYNTHETIC.evaluate_fraction("2**5-3"), Fraction(29))
        with self.assertRaises(ValueError):
            SYNTHETIC.evaluate_fraction("__import__('os').system('true')")

    def test_code_block_validation(self):
        self.assertEqual(SYNTHETIC.validate_code_blocks("```python\nx = 1\n```"), [])
        self.assertIn("invalid_python_block", SYNTHETIC.validate_code_blocks("```python\nx =\n```"))
        self.assertIn("unsupported_code_fence", SYNTHETIC.validate_code_blocks("```javascript\nlet x=1;\n```"))

    def test_math_checks_are_verified_and_present(self):
        base = "数" * 1500 + "\n自动校验附录\n(2+3)*4=20\n1/2+1/3=5/6\n2**5-3=29"
        annotation = {
            "domain": "math_zh",
            "text": base,
            "verification_cases": [
                {"expression": "(2+3)*4", "expected": "20"},
                {"expression": "1/2+1/3", "expected": "5/6"},
                {"expression": "2**5-3", "expected": "29"},
            ],
        }
        self.assertEqual(SYNTHETIC.validate_generation(annotation), [])
        annotation["verification_cases"][0]["expected"] = "21"
        self.assertIn("wrong_math_check", SYNTHETIC.validate_generation(annotation))


if __name__ == "__main__":
    unittest.main()
