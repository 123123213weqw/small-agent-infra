import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
p=Path(__file__).resolve().parents[1]/'scripts/p0_download_supply_hfmirror_v4.py'
s=importlib.util.spec_from_file_location('mirror_download',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class DownloadTests(unittest.TestCase):
    def setUp(self):self.t=tempfile.TemporaryDirectory();self.root=Path(self.t.name)
    def tearDown(self):self.t.cleanup()
    def item(self,data=b'PAR1fixture'):
        return {'source_id':'fixture','dataset':'owner/dataset','revision':'a'*40,'path':'data/a.parquet',
            'destination':str(self.root/'raw/a.parquet'),'bytes':len(data),'sha256':__import__('hashlib').sha256(data).hexdigest(),'action':'download'}
    def fake(self,data):
        def run(argv,check):Path(argv[argv.index('--output')+1]).write_bytes(data)
        return run
    def test_pinned_mirror_url(self):self.assertIn('hf-mirror.com/datasets/owner/dataset/resolve/'+'a'*40,m.url(self.item()))
    def test_new_file_published_only_after_hash_verification(self):
        i=self.item();r=m.fetch(i,self.root/'state',0,runner=self.fake(b'PAR1fixture'))
        self.assertFalse(r['training_approved']);self.assertTrue(Path(i['destination']).exists())
    def test_wrong_hash_not_published_and_partial_preserved(self):
        i=self.item()
        with self.assertRaises(ValueError):m.fetch(i,self.root/'state',0,runner=self.fake(b'WRONG'))
        self.assertFalse(Path(i['destination']).exists());self.assertTrue(Path(i['destination']+'.part').exists())
    def test_existing_valid_file_not_downloaded_again(self):
        i=self.item();p=Path(i['destination']);p.parent.mkdir();p.write_bytes(b'PAR1fixture')
        def forbidden(*a,**kw):self.fail('duplicate network request')
        m.fetch(i,self.root/'state',0,runner=forbidden)
    def test_missing_reuse_file_fails(self):
        i=self.item();i['action']='reuse'
        with self.assertRaises(FileNotFoundError):m.fetch(i,self.root/'state',0)
    def test_finished_partial_verified_without_network(self):
        i=self.item();p=Path(i['destination']+'.part');p.parent.mkdir();p.write_bytes(b'PAR1fixture')
        def forbidden(*a,**kw):self.fail('unnecessary network request')
        m.fetch(i,self.root/'state',0,runner=forbidden);self.assertTrue(Path(i['destination']).exists())
    def test_traversal_rejected(self):
        i=self.item();i['path']='../a.parquet'
        with self.assertRaises(ValueError):m.url(i)
    def test_duplicate_destination_rejected(self):
        i=self.item()
        with self.assertRaises(ValueError):m.validate({'stage':'raw_download_only_not_training_approved','allowed_root':str(self.root),'items':[i,i]})
    def test_disk_reserve_blocks_download(self):
        with self.assertRaises(RuntimeError):m.fetch(self.item(),self.root/'state',10**30)

if __name__=='__main__':unittest.main()
