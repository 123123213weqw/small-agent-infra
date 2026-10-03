"""CPU equivalence/fail-closed fixtures only, not corpus approval."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import p0_parallel_cpu_v13 as parallel
import p0_inventory_supply_v15 as queue
from small_agent.curation.supply_control import file_sha
from small_agent.curation.release_gates_v5 import lexical, sha


class ParallelTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.r = Path(self.tmp.name)
        texts = ['def f%d(x):\n    return x + %d\n# 中文 数学 α + β\n' % (i, i) for i in range(70)]
        tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=400,
            special_tokens=['<s%d>' % i for i in range(15)], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        self.tokenizer = self.r / 'tokenizer.json'; tok.save(str(self.tokenizer)); self.tok = tok
        self.rows = [dict(text=t, normalized_sha256=sha(t), own_text_tokens=len(tok.encode(t).ids),
                          split='train' if i<66 else ('validation' if i<68 else 'test'), domain='code')
                     for i,t in enumerate(texts)]
        self.source = self.r / 'source.parquet'; pq.write_table(pa.Table.from_pylist(self.rows), self.source, row_group_size=19)

    def test_parallel_encode_is_byte_identical_and_ordered(self):
        sequential = list(parallel.encoded_documents(self.source,self.tokenizer,1))
        self.assertEqual(sequential,list(parallel.encoded_documents(self.source,self.tokenizer,2)))
        self.assertEqual([d.document_id for d in sequential],[r['normalized_sha256'] for r in self.rows])

    def test_bad_count_fails_closed(self):
        import pyarrow as pa, pyarrow.parquet as pq
        self.rows[-1]['own_text_tokens'] += 1
        pq.write_table(pa.Table.from_pylist(self.rows),self.source)
        with self.assertRaisesRegex(ValueError,'Token count'):
            list(parallel.encoded_documents(self.source,self.tokenizer,2))

    def test_full_decode_reencode_and_corruption_rejected(self):
        import numpy as np
        items=[(r['text'],np.asarray(self.tok.encode(r['text']).ids,dtype='<u2').tobytes()) for r in self.rows]
        with parallel.CodecChecks(self.tokenizer,2) as checks:
            for text,data in items:checks.add(text,data)
            self.assertEqual(checks.finish(),len(items))
        with self.assertRaisesRegex(ValueError,'Packed document'):
            with parallel.CodecChecks(self.tokenizer,2) as checks:
                for text,data in items:checks.add(text+' changed',data)

    def test_parallel_sketches_equal_original_exact_algorithm(self):
        import heapq, pyarrow.parquet as pq
        e=dict(path=str(self.source),sha256=file_sha(self.source))
        m=parallel.build_sketch(self.source,e,self.r/'sketch',2)
        rows=pq.read_table(m['path']).to_pylist()
        for i,(a,b) in enumerate(zip(rows,self.rows)):
            values=lexical(b['text'])
            self.assertEqual(a,dict(body_sha256=b['normalized_sha256'],split=b['split'],row=i,
                                   shingle_count=len(values),sketch=heapq.nsmallest(64,values)))
        self.assertEqual(m,parallel.build_sketch(self.source,e,self.r/'sketch',2))

    def test_corrupt_sketch_and_changed_source_rejected(self):
        e=dict(path=str(self.source),sha256=file_sha(self.source))
        m=parallel.build_sketch(self.source,e,self.r/'sketch',1)
        Path(m['path']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'cache changed'):parallel.build_sketch(self.source,e,self.r/'sketch',2)
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'source changed'):parallel.build_sketch(self.source,e,self.r/'other',2)

    def test_pending_work_is_bounded(self):
        from concurrent.futures import Future
        consumed=[]
        def inputs():
            for i in range(100):consumed.append(i);yield i
        class Pool:
            def submit(self,fn,item):f=Future();f.set_result(fn(item));return f
        values=parallel.ordered_map(Pool(),abs,inputs(),4)
        self.assertEqual(next(values),0);self.assertEqual(len(consumed),4)
        self.assertEqual(list(values),list(range(1,100)))

    def test_parallel_near_global_local_decisions_equal_frozen_v6(self):
        import sqlite3
        from small_agent.curation.stock_v6 import DiskNearIndex
        from p0_parallel_near_v15 import ParallelNearIndex
        ledger=self.r/'ledger.sqlite'
        with sqlite3.connect(ledger) as db:
            db.execute('CREATE TABLE documents(body_sha TEXT,split TEXT,domain TEXT,tokens INTEGER)')
            db.execute('CREATE TABLE releases(batch_id TEXT)')
            db.executemany('INSERT INTO documents VALUES (?,?,?,?)',[(r['normalized_sha256'],r['split'],r['domain'],r['own_text_tokens']) for r in self.rows])
        legacy=self.r/'legacy.json';legacy.write_text(json.dumps(dict(evidence_files=[dict(path=str(self.source),sha256=file_sha(self.source))])))
        idx=DiskNearIndex(self.r/'index');self.addCleanup(idx.close)
        manifest=idx.update(ledger,legacy,self.r/'cache')
        rows=[dict(r,text=r['text']+'# comment\n',normalized_sha256=sha(r['text']+'# comment\n')) for r in self.rows]
        fresh='Unique words in a brand new document with no program syntax. '+('distinct mathematical reasoning '*20)
        rows += [dict(self.rows[0],text=fresh,normalized_sha256=sha(fresh)),dict(self.rows[0],text=fresh+'\n',normalized_sha256=sha(fresh+'\n'))]
        policy=dict(threshold=.9,sketch_size=64,minimum_shared_fraction=.5,maximum_pair_checks_per_document=2000)
        expected=idx.filter(rows,manifest,policy,ledger)
        optimized=ParallelNearIndex(self.r/'index');self.addCleanup(optimized.close)
        self.assertEqual(expected,optimized.filter(rows,manifest,policy,ledger,2))
        self.assertTrue(any(r['reason']=='local_near_duplicate' for r in expected[1]['exclusions']))
        dense=dict(policy,maximum_pair_checks_per_document=0)
        self.assertEqual(idx.filter(rows,manifest,dense,ledger),optimized.filter(rows,manifest,dense,ledger,2))
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Reference changed'):optimized.filter(rows,manifest,policy,ledger,2)

    def test_inflight_v11_release_keeps_original_engine(self):
        out=self.r/'release';out.mkdir();args=['--output',str(out)]
        self.assertEqual(queue.publisher_for(args),'p0_release_inventory_batch_v15.py')
        (out/'INPUT_BINDING.json').write_text(json.dumps(dict(engine_sha256={'/old/p0_release_inventory_batch_v11.py':'hash'})))
        self.assertEqual(queue.publisher_for(args),'p0_release_inventory_batch_v11.py')

    def test_v13_full_pipeline_atomic_recovery_all_domains(self):
        import test_full_inventory_v11 as oldtests
        import p0_release_inventory_batch_v15 as pub
        perf=self.r/'performance.json';perf.write_text(json.dumps(dict(release_cpu_workers=2)))
        with patch.dict(os.environ,P0_PARALLEL_CONFIG=str(perf)),patch.object(oldtests,'publisher',pub):
            oldtests.InventoryTests().test_generic_real_pack_recovery_all_domains()


if __name__=='__main__':unittest.main()
