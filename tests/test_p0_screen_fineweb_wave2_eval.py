import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from p0_screen_fineweb_wave2_eval import AnchorAutomaton, words  # noqa: E402


class ExactPhraseScreenTest(unittest.TestCase):
    def test_requires_complete_contiguous_phrase(self):
        screen = AnchorAutomaton([
            ("arc", "item-1", words("the cat sat on the mat")),
            ("hellaswag", "item-2", words("cat sat on the mat again")),
        ])
        self.assertEqual(
            list(screen.find(words("The cat sat on the mat again."))),
            [("arc", "item-1", 6), ("hellaswag", "item-2", 6)],
        )
        self.assertEqual(list(screen.find(words("the cat sat on a mat again"))), [])

    def test_full_phrase_must_match_beyond_eight_word_anchor(self):
        phrase = words("one two three four five six seven eight nine ten")
        screen = AnchorAutomaton([("arc", "long", phrase)])
        self.assertEqual(list(screen.find(words("one two three four five six seven eight nine X"))), [])
        self.assertEqual(list(screen.find(words("one two three four five six seven eight nine ten"))),
                         [("arc", "long", 10)])

    def test_repeated_occurrences_report_one_item_per_document(self):
        screen = AnchorAutomaton([("arc", "repeat", words("one two three four five six"))])
        self.assertEqual(len(list(screen.find(words("one two three four five six; " * 3)))), 1)


if __name__ == "__main__":
    unittest.main()
