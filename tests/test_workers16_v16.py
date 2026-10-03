"""16-worker equivalence and fail-closed checks; not corpus quality certification."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
from test_parallel_cpu_v13 import ParallelTests
import p0_parallel_cpu_v13 as parallel


class Workers16Tests(unittest.TestCase):
    setUp = ParallelTests.setUp

    def test_encoded_documents_8_16_equal(self):
        self.assertEqual(list(parallel.encoded_documents(self.source, self.tokenizer, 8)),
                         list(parallel.encoded_documents(self.source, self.tokenizer, 16)))

    def test_full_codec_and_corruption_fail_closed_16(self):
        import numpy as np
        items = [(r['text'], np.asarray(self.tok.encode(r['text']).ids, dtype='<u2').tobytes())
                 for r in self.rows]
        with parallel.CodecChecks(self.tokenizer, 16) as checks:
            for text, data in items:
                checks.add(text, data)
            self.assertEqual(checks.finish(), len(items))
        with self.assertRaisesRegex(ValueError, 'Packed document'):
            with parallel.CodecChecks(self.tokenizer, 16) as checks:
                for text, data in items:
                    checks.add(text + ' changed', data)

    def test_global_local_dense_hold_and_changed_reference_16(self):
        from p0_parallel_near_v13 import ParallelNearIndex
        original = ParallelNearIndex.filter

        def sixteen(index, rows, manifest, policy, database, workers):
            return original(index, rows, manifest, policy, database, 16)

        with patch.object(ParallelNearIndex, 'filter', sixteen):
            ParallelTests.test_parallel_near_global_local_decisions_equal_frozen_v6(self)

    def test_atomic_release_recovery_all_domains_16(self):
        import test_full_inventory_v11 as fixtures
        import p0_release_inventory_batch_v15 as publisher
        config = self.r / 'performance16.json'
        config.write_text(json.dumps(dict(release_cpu_workers=16)))
        with patch.dict(os.environ, P0_PARALLEL_CONFIG=str(config)), \
                patch.object(fixtures, 'publisher', publisher):
            fixtures.InventoryTests().test_generic_real_pack_recovery_all_domains()


if __name__ == '__main__':
    unittest.main()
