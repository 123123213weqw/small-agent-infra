"""Engineering fixtures, not real-corpus quality certification."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import p0_training_bundle_v5 as bundle
from p0_enqueue_calibrated_python_v5 import eligible_reason, partition
from small_agent.curation.release_gates_v5 import sha
from small_agent.curation.supply_control import file_sha
from small_agent.data.sharded_pack import TokenizedDocument, pack_sharded


class ProducerTests(unittest.TestCase):
    def row(self):
        return dict(text='print(1)', normalized_sha256=sha('print(1)'), source_id='fixture',
                    language='Python', own_text_tokens=100, record_license='mit',
                    policy_bucket='record_license_whitelisted', split='train', family_id='a')

    def reason(self, **updates):
        r = self.row(); r.update(updates)
        return eligible_reason(r, {'licenses': ['mit']},
                    {'source_id': 'fixture', 'approved_strata': ['short/python', 'medium/python']}, set(), {})

    def test_registered_short_python(self): self.assertIsNone(self.reason())
    def test_other_language_hold(self): self.assertIn('uncalibrated', self.reason(language='Java'))
    def test_long_hold(self): self.assertIn('uncalibrated', self.reason(own_text_tokens=9000))
    def test_missing_license_hold(self): self.assertEqual(self.reason(record_license='unknown'), 'license_hold')
    def test_changed_body_fails(self):
        with self.assertRaises(ValueError): self.reason(text='print(2)')
    def test_family_conflict_hold(self):
        r = self.row()
        self.assertEqual(eligible_reason(r, {'licenses': ['mit']},
            {'source_id': 'fixture', 'approved_strata': ['short/python']}, set(), {'a': 'test'}), 'released_family_split_hold')
    def test_known_bad_hold(self):
        r = self.row()
        self.assertEqual(eligible_reason(r, {'licenses': ['mit']},
            {'source_id': 'fixture', 'approved_strata': ['short/python']}, {r['normalized_sha256']}, {}), 'known_audit_bad_hold')
    def test_split_is_preserved_and_small_tail_merged(self):
        train = [dict(split='train', own_text_tokens=2100, id=i) for i in range(5)]
        val = [dict(split='validation', id='v' + str(i)) for i in range(2)]
        test = [dict(split='test', id='t' + str(i)) for i in range(2)]
        parts = partition(train, val, test, 5000)
        self.assertEqual(len(parts), 2)
        self.assertEqual([r for part in parts for r in part if r['split']=='train'], train)
        self.assertEqual(sum(r['split']=='validation' for part in parts for r in part), 2)
    def test_no_manufactured_validation(self):
        with self.assertRaises(ValueError): partition([dict(own_text_tokens=5000)], [], [], 5000)


class BundleTests(unittest.TestCase):
    def fixture(self, root):
        entries = []
        for i in range(2):
            run = root / f'run{i}'; run.mkdir()
            docs = [dict(body_sha256=sha(f'{i}-{split}'), family_id=f'{i}-{split}',
                         split=split, domain='code', body_tokens=500 if split=='train' else 30)
                    for split in ('train', 'validation', 'test')]
            idx = run / 'documents.jsonl'; idx.write_text(''.join(json.dumps(r)+'\n' for r in docs))
            pack_sharded([TokenizedDocument(d['body_sha256'], d['split'], 'code', [3+i]*d['body_tokens']) for d in docs],
                         run/'tokenized', sequence_length=128, target_shard_tokens=1024,
                         vocab_size=32768, eos_token_id=2, tokenizer_sha256='f'*64,
                         mixture={'code': 1}, license_gate={'training_eligible':True}, source_manifest_sha256='e'*64)
            ready = run/'READY_INTERNAL.json'
            ready.write_text(json.dumps(dict(document_index=dict(path=str(idx), sha256=file_sha(idx)))))
            entries.append(dict(batch_id=str(i), manifest=str(run/'tokenized/manifest.json'),
                ready_receipt=str(ready), train_body_tokens=500))
        return dict(packs=entries, unique_train_body_tokens=1000, training_started=False)

    def test_same_domain_hardlink_reader_and_replay(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t); c=self.fixture(r)
            with patch.object(bundle, 'catalog', return_value=c):
                ready=bundle.build(r/'unused.sqlite',r/'bundles')
                generation=Path(ready['data_dir'])
                self.assertEqual(ready['unique_train_body_tokens'],1000)
                self.assertFalse(ready['training_started'])
                m=json.loads((generation/'manifest.json').read_text())
                self.assertEqual(set(m['mixture']),{'code'})
                self.assertEqual((r/'run0/tokenized/train-00000.bin').stat().st_ino,
                                 (generation/'train-00000.bin').stat().st_ino)
                self.assertEqual(bundle.build(r/'unused.sqlite',r/'bundles'),ready)
                self.assertEqual(len(list((r/'bundles').glob('*/manifest.json'))),1)
    def test_empty_catalog_fails(self):
        with tempfile.TemporaryDirectory() as t, patch.object(bundle,'catalog',return_value={'packs':[]}):
            with self.assertRaises(ValueError):bundle.build(Path(t)/'db',Path(t)/'bundles')
    def test_cross_pack_heldout_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);c=self.fixture(r);c['packs'][1]['ready_receipt']=c['packs'][0]['ready_receipt']
            with self.assertRaises(ValueError):bundle.check_documents(c['packs'])
    def test_mutated_hardlink_not_reused(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);c=self.fixture(r)
            with patch.object(bundle,'catalog',return_value=c):
                ready=bundle.build(r/'db',r/'bundles')
                (Path(ready['data_dir'])/'train-00000.bin').write_bytes(b'changed')
                with self.assertRaises(ValueError):bundle.build(r/'db',r/'bundles')


if __name__=='__main__':unittest.main()
