import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
p=Path(__file__).resolve().parents[1]/'scripts/p0_curation_deepseek_env_v3.py'
s=importlib.util.spec_from_file_location('loader',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ProfileTests(unittest.TestCase):
    def profile(self,root):
        names=['scripts/p0_curation_worker_v3.py','scripts/p0_curation_v3.py',
               'src/small_agent/curation/core.py','src/small_agent/curation/store.py','src/small_agent/curation/ledger.py']
        files={}
        for name in names:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
            files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        credential=root/'credentials.env';credential.write_text('DEEPSEEK_API_KEY=fixture-test-only\n');credential.chmod(0o600)
        x={'version':1,'api_base':'https://api.deepseek.com','model':'deepseek-flash',
           'worker_code_root':str(root),'worker_files_sha256':files,'credentials_env':str(credential),
           'expected_key_sha256':hashlib.sha256(b'fixture-test-only').hexdigest(),
           'campaign_budget_cny':0,'campaign_ledger':str(root/'ledger.sqlite'),'max_output_tokens':2048}
        p=root/'profile.json';p.write_text(json.dumps(x));return p,x
    def test_profile_reuses_existing_worker(self):
        with tempfile.TemporaryDirectory() as d:
            p,x=self.profile(Path(d));profile,key,worker=m.load_profile(p)
            self.assertEqual(key,'fixture-test-only');self.assertEqual(worker,Path(d).resolve()/'scripts/p0_curation_worker_v3.py')
    def test_wrong_account_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p,x=self.profile(Path(d));Path(x['credentials_env']).write_text('DEEPSEEK_API_KEY=different-fixture\n')
            with self.assertRaises(ValueError):m.load_profile(p)
    def test_changed_engine_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p,x=self.profile(Path(d));(Path(d).resolve()/'scripts/p0_curation_worker_v3.py').write_text('changed')
            with self.assertRaises(ValueError):m.load_profile(p)
    def test_zero_budget_cannot_send(self):
        with tempfile.TemporaryDirectory() as d:
            p,x=self.profile(Path(d))
            with self.assertRaises(ValueError):m.profile_worker_args(x,['--budget-cny','1','--allow-api'])
    def test_nonofficial_endpoint_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p,x=self.profile(Path(d));x['api_base']='https://example.org';p.write_text(json.dumps(x))
            with self.assertRaises(ValueError):m.load_profile(p)
    def test_world_readable_credential_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p,x=self.profile(Path(d));Path(x['credentials_env']).chmod(0o644)
            with self.assertRaises(ValueError):m.load_profile(p)
    def test_duplicate_budget_flag_stops(self):
        with self.assertRaises(ValueError):m.argument(['--budget-cny','1','--budget-cny','50'],'--budget-cny')
if __name__=='__main__':unittest.main()
