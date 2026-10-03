"""Unchanged worker/cursor gates plus a slow-copy scheduling regression."""
import json
from pathlib import Path
import threading
from unittest.mock import patch
import unittest
import test_distributed_scan_v12 as original
import p0_distributed_scan_v14 as optimized
from small_agent.curation.supply_control import digest,file_sha

class DistributedOverlapTests(original.DistributedTests):
    def setUp(self):
        super().setUp()
        p=patch.object(original,'coord',optimized);p.start();self.addCleanup(p.stop)

    def test_ready_cpu_runs_before_slow_remote_copy_completes(self):
        db=self.db();root=self.r/'new'
        reg=dict(files=1,sources=[dict(self.source,files=[self.file])]);(root/'REGISTRY.json').write_text(json.dumps(reg))
        binding=digest(dict(registry=file_sha(root/'REGISTRY.json'),adapter=file_sha(optimized.ROOT/optimized.CODE[1]),scanner=file_sha(optimized.ROOT/'scripts/p0_full_inventory_scan_v11.py')))
        db.execute("INSERT INTO meta VALUES ('scan_identity',?)",(binding,));db.commit()
        ready=threading.Event();began=[]
        primary=dict(id='primary',workers=1,root=str(self.r/'node'),python='unused')
        remote=dict(id='slow_remote',workers=1,root=str(self.r/'remote'),python='unused',ssh='unused')
        cfg=dict(self.cfg,root=str(root),nodes=[primary,remote],range_rows=2)
        def prepare(node,cfg,file):
            if node.get('ssh') and not ready.wait(10):raise RuntimeError('Primary never started while copy pending')
            return node
        def execute(node,cfg,task):
            if not began:
                began.append(node['id']);self.assertFalse(ready.is_set());ready.set()
            out=Path(cfg['distributed_root'])/'results'/original.worker.identity(task)/'output'
            actual=dict(task,local_file=self.file['path'],local_tokenizer=str(self.token))
            receipt=original.worker.run(actual,out)
            return dict(node=node['id'],rows=receipt['rows'],elapsed_seconds=receipt['elapsed_seconds'])
        with patch.object(optimized,'prepare',side_effect=prepare),patch.object(optimized,'execute',side_effect=execute),patch.object(optimized,'shell',return_value=''):
            optimized.run(cfg)
        self.assertEqual(began,['primary'])
        self.assertEqual(db.execute('SELECT next_pos FROM files').fetchone()[0],len(self.rows))
        self.assertEqual(db.execute('SELECT state FROM files').fetchone()[0],'complete')

if __name__=='__main__':unittest.main()
