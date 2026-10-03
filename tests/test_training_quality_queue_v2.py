import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import training_eval_queue_v1 as queue
import prepare_training_quality_holdout_v2 as panel


class QueueTests(unittest.TestCase):
    def setup_tree(self,root):
        run=root/'training';entry=run/'evaluation_requests/step_00002000';cp=entry/'checkpoint'
        cp.mkdir(parents=True);(cp/'model.safetensors').write_bytes(b'weights');(cp/'manifest.json').write_text('{"step":2000}')
        request={'training_run':str(run.resolve()),'eval_only_not_resumable':True,'step':2000,'checks':['generation','abilities_pilot','code_pilot'],'checkpoint_manifest_sha256':queue.sha(cp/'manifest.json')}
        (entry/'request.json').write_text(json.dumps(request));(run/'status.json').write_text('{"state":"completed"}')
        eroot=root/'eval';eroot.mkdir();cfg=root/'config.json';cfg.write_text('{}');prompts=root/'prompts.json';prompts.write_text('{}')
        argv=['queue','--run',str(run),'--eval-root',str(eroot),'--base-root',str(root/'base'),'--config',str(cfg),'--prompts',str(prompts),'--gpus','0','--once','--outbox']
        return entry,cp,argv

    def test_consumes_request_and_releases_only_eval_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entry,cp,argv=self.setup_tree(root)
            formal=root/'formal_model';formal.write_bytes(b'original')
            with mock.patch.object(sys,'argv',argv),mock.patch.object(queue,'free_gpus',return_value=True),mock.patch.object(queue,'dispatch') as dispatch:
                queue.main()
            self.assertEqual(dispatch.call_args.kwargs['request']['checks'],['generation','abilities_pilot','code_pilot'])
            self.assertFalse((cp/'model.safetensors').exists());self.assertTrue((entry/'COMPLETE.json').exists())
            self.assertEqual(formal.read_bytes(),b'original')

    def test_failed_scoring_keeps_pin_and_does_not_mark_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            entry,cp,argv=self.setup_tree(Path(directory))
            with mock.patch.object(sys,'argv',argv),mock.patch.object(queue,'free_gpus',return_value=True),mock.patch.object(queue,'dispatch',side_effect=RuntimeError('test')):
                with self.assertRaises(RuntimeError):queue.main()
            self.assertTrue((cp/'model.safetensors').exists());self.assertFalse((entry/'COMPLETE.json').exists())

    def test_occupied_card_does_not_launch_or_kill_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            entry,cp,argv=self.setup_tree(Path(directory))
            with mock.patch.object(sys,'argv',argv),mock.patch.object(queue,'free_gpus',return_value=False),mock.patch.object(queue,'dispatch') as dispatch:
                with self.assertRaises(RuntimeError):queue.main()
            dispatch.assert_not_called();self.assertTrue((cp/'model.safetensors').exists())

    def test_wrong_manifest_refused_before_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            entry,cp,argv=self.setup_tree(Path(directory));(cp/'manifest.json').write_text('{"step":3000}')
            with mock.patch.object(sys,'argv',argv),mock.patch.object(queue,'free_gpus',return_value=True),mock.patch.object(queue,'dispatch') as dispatch:
                with self.assertRaises(ValueError):queue.main()
            dispatch.assert_not_called()

    def test_exact_window_hash_and_chunk_boundary(self):
        try:import numpy as np
        except ImportError:self.skipTest('numpy required in remote environment')
        values=np.arange(100,dtype=np.uint16)
        hashes=panel.window_hashes(values,width=32)
        h=0
        for value in values[:32]:h=(h*1000003+int(value)+1)%(1<<64)
        self.assertEqual(int(hashes[0]),h)
        # All window start positions appear once with width-1 overlap.
        parts=[panel.window_hashes(values[base:min(len(values),base+17+31)]) for base in range(0,len(values)-31,17)]
        self.assertTrue(np.array_equal(hashes,np.concatenate(parts)))

    def test_dispatch_freezes_prompts_and_refuses_changed_completed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entry,cp,argv=self.setup_tree(root)
            run=root/'training';eroot=root/'eval';(eroot/'runs').mkdir()
            config=root/'config.json';prompts=root/'prompts.json'
            training_config=root/'training_config.json';training_config.write_text('{"max_steps":2000}')
            (run/'run_manifest.json').write_text(json.dumps({'run_config':{'path':str(training_config)},'model_spec':{'path':str(root/'code/configs/model/spec.json')}}))
            args=SimpleNamespace(run=run,eval_root=eroot,base_root=root/'base',config=config,prompts=prompts,gpus=[0],once=False,scope='pilot')
            with mock.patch.object(queue.subprocess,'Popen') as popen:
                popen.return_value.wait.return_value=0
                queue.dispatch(args,cp,{},request={'checks':['generation']})
            output=eroot/'runs/training_queue_step_00002000_scheduled'
            self.assertEqual((output/'prompts.json').read_bytes(),prompts.read_bytes())
            self.assertEqual((output/'eval_config.json').read_bytes(),config.read_bytes())
            for call in popen.call_args_list:
                self.assertIn(str(output/'prompts.json'),call.args[0])
            prompts.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'inputs changed'):
                queue.dispatch(args,cp,{},request={'checks':['generation']})


if __name__=='__main__':unittest.main()
