"""Physical five-domain engineering fixtures, not genuine approved training inventory."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import p0_balanced_training_bundle_v9 as bundle
from small_agent.curation.supply_control import file_sha
from small_agent.curation.release_gates_v5 import sha
from small_agent.data.sharded_pack import TokenizedDocument,pack_sharded
RECIPE=ROOT/'configs/curation/base_1b_supply_20b_v4.json'


class BalancedBundleTests(unittest.TestCase):
    def fixture(self,root,domains):
        packs=[];total=0
        for i,domain in enumerate(domains):
            run=root/domain;run.mkdir();docs=[dict(body_sha256=sha(domain+split),family_id=domain+split,split=split,domain=domain,body_tokens=10000+i*123 if split=='train' else 500) for split in ('train','validation','test')]
            idx=run/'documents.jsonl';idx.write_text(''.join(json.dumps(d)+'\n' for d in docs))
            pack_sharded([TokenizedDocument(d['body_sha256'],d['split'],domain,[3+i]*d['body_tokens']) for d in docs],run/'tokenized',sequence_length=128,target_shard_tokens=20000,vocab_size=32768,eos_token_id=2,tokenizer_sha256='f'*64,mixture={domain:1},license_gate={'training_eligible':True},source_manifest_sha256='e'*64)
            ready=run/'READY_INTERNAL.json';ready.write_text(json.dumps(dict(document_index=dict(path=str(idx),sha256=file_sha(idx)))))
            n=docs[0]['body_tokens'];total+=n;packs.append(dict(batch_id=domain,manifest=str(run/'tokenized/manifest.json'),ready_receipt=str(ready),train_body_tokens=n))
        return dict(packs=packs,unique_train_body_tokens=total,training_started=False)
    def test_missing_domain_never_renormalizes_or_publishes_bundle(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);c=self.fixture(r,['code'])
            with patch.object(bundle,'catalog',return_value=c):a=bundle.build(r/'db',r/'bundles',RECIPE)
            self.assertEqual(a['status'],'WAITING_MISSING_DOMAINS_NO_RENORMALIZATION')
            self.assertFalse((r/'bundles/CURRENT').exists())
    def test_five_domains_physical_reader_and_frozen_recipe_replay(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);c=self.fixture(r,list(json.loads(RECIPE.read_text())['targets_by_domain']))
            with patch.object(bundle,'catalog',return_value=c):
                a=bundle.build(r/'db',r/'bundles',RECIPE);b=bundle.build(r/'db',r/'bundles',RECIPE)
            self.assertEqual(a,b);self.assertEqual(a['trainer_binding']['mixture_block_size'],10000)
            m=json.loads((Path(a['data_dir'])/'manifest.json').read_text());target=m['target_loss_token_weights']
            self.assertEqual(target['math'],.1);self.assertEqual(target['general_en'],.35)
            ratios={d:m['mixture'][d]*m['domain_mean_valid_loss_tokens'][d] for d in target};norm=sum(ratios.values())
            for d in target:self.assertAlmostEqual(ratios[d]/norm,target[d])
            self.assertFalse(a['training_started'])
if __name__=='__main__':unittest.main()
