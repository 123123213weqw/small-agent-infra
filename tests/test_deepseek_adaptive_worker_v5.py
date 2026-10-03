"""Offline concurrency/billing tests. Fixtures do not certify real data quality."""
import importlib.util,json,sys,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
import p0_deepseek_adaptive_worker_v5 as adaptive
from test_source_audit_v4 import doc,label,SCHEMA,POLICY,REGISTRY
from small_agent.curation.core import canonical,sha
from small_agent.curation.store import prepare,Batch

class AdaptiveTests(unittest.TestCase):
 def setup_batch(self,root,n=20):
  docs=[]
  for i in range(n):
   d=doc();d['document_id']=sha(str(i));d['document_text']+=f'\n# Document number {i}\n';d['body_sha256']=sha(d['document_text']);d['document_id']=d['body_sha256'];docs.append(d)
  src=root/'input.jsonl';src.write_text(''.join(canonical({k:v for k,v in d.items() if k not in ['review_payload','lint_flags','body_sha256']})+'\n' for d in docs))
  for name,obj in [('policy.json',POLICY),('schema.json',SCHEMA),('registry.json',REGISTRY)]: (root/name).write_text(json.dumps(obj))
  prepare(src,root/'batch',root/'policy.json',root/'schema.json',ROOT/'configs/curation/p0_source_audit_prompt_v4.txt',root/'registry.json')
  b=Batch(root/'batch');key='fixture-key';p=dict(campaign_ledger=str(root/'campaign.sqlite'),campaign_budget_cny=1,expected_key_sha256=sha(key),model='deepseek-flash',api_base='https://api.deepseek.com',max_output_tokens=2048,pricing_cny_per_million=REGISTRY['judges']['deepseek']['endpoint']['pricing_cny_per_million'])
  return b,p,key
 def responder(self,body):
  payload=json.loads(body['input'][0]['content']);d=dict(document_id=payload['document_id'],body_sha256=payload['body_sha256'],document_text=payload['document_text']);a=label(d)
  return dict(status='completed',id='fixture-response',model='deepseek-flash',usage=dict(input_tokens=100,output_tokens=20),output=[dict(type='message',content=[dict(type='output_text',text=canonical(a))])])
 def test_ramp_and_rate_backoff(self):
  self.assertEqual(adaptive.next_width(8,32,[dict(status='ok')],1,1),(16,'ramp'))
  self.assertEqual(adaptive.next_width(16,32,[dict(http_status=429)],1,1),(8,'rate_backoff'))
  self.assertEqual(adaptive.next_width(16,32,[dict(http_status=503)],1,1),(8,'rate_backoff'))
  self.assertEqual(adaptive.next_width(16,32,[],80,1),(8,'latency_backoff'))
 def test_real_thread_overlap_and_idempotent_no_second_billing(self):
  with tempfile.TemporaryDirectory() as t:
   b,p,k=self.setup_batch(Path(t));active=0;peak=0;lock=threading.Lock()
   def send(url,kind,body,key,timeout):
    nonlocal active,peak
    with lock:active+=1;peak=max(peak,active)
    time.sleep(.025)
    result=self.responder(body)
    with lock:active-=1
    return result
   try:
    with patch.object(adaptive.old,'send',side_effect=send) as mock:
     r=adaptive.run(b,p,k,initial=4,maximum=8);self.assertEqual(r['stop_reason'],None);self.assertEqual(r['counts']['new_api_requests'],20);self.assertGreater(peak,1);self.assertLessEqual(peak,8)
     r2=adaptive.run(b,p,k,initial=4,maximum=8);self.assertEqual(r2['selected'],0);self.assertEqual(mock.call_count,20)
   finally:b.close()
 def test_timeout_stops_future_windows_and_replay_does_not_send(self):
  with tempfile.TemporaryDirectory() as t:
   b,p,k=self.setup_batch(Path(t))
   try:
    with patch.object(adaptive.old,'send',side_effect=TimeoutError) as mock:
     r=adaptive.run(b,p,k,initial=8,maximum=32);self.assertEqual(r['stop_reason'],'unknown_transport_or_fatal_response');self.assertEqual(mock.call_count,1)
     # Run same failed batch again: queued future jobs must not consume budget without inspection.
     # Durable unknown/fatal receipt poisons admission, not just a pending SQLite row.
     with self.assertRaises(ValueError):adaptive.run(b,p,k,initial=8,maximum=32)
     self.assertEqual(mock.call_count,1)
   finally:b.close()
 def test_invalid_concurrency(self):
  with self.assertRaises(ValueError):adaptive.run(None,None,None,initial=8,maximum=2)
 def test_model_mismatch_is_not_safe_known_invalid(self):
  from small_agent.curation.store import protocol
  with tempfile.TemporaryDirectory() as t:
   b,p,k=self.setup_batch(Path(t),1)
   try:
    d=next(iter(b.docs.values()));j=b.registry['judges']['deepseek'];base=protocol(d,b.prompt,b.schema,j);body=adaptive.old.request_body(d,b.prompt,b.schema,j)
    response=self.responder(body);response['model']='unapproved-model'
    with patch.object(adaptive.old,'send',return_value=response):_,r=adaptive.attempt(('x',d,body,base),j,b.prompt,b.schema,k,10)
    self.assertFalse(adaptive.safe_known_result(r,base,j['served_models']))
   finally:b.close()
if __name__=='__main__':unittest.main()

class ExpandedControlBoundTests(unittest.TestCase):
 def test_limit_and_concurrency_never_unbounded(self):
  from p0_deepseek_adaptive_worker_v5 import run
  for params in [dict(limit=10001),dict(maximum=2501),dict(initial=5,maximum=4),dict(limit=0)]:
   with self.assertRaises(ValueError):run(None,None,'',**params)
