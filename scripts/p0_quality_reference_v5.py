#!/usr/bin/env python3
"""Test-verified external Python references + explicitly marked corruption controls.

Benchmark text remains evaluation-only. The controls measure narrow format/test
calibration, NOT an independently sampled natural-corpus factual defect rate.
Only checksum-pinned, import-free MBPP programs passing restricted AST admission
are tested. This is not a general sandbox for untrusted corpus execution.
"""
import argparse,ast,builtins,json,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import sha,write
from small_agent.curation.supply_control import file_sha
SAFE=['len','range','list','tuple','set','dict','zip','enumerate','sum','max','min','sorted','reversed','abs','int','float','bool','str','chr','ord','round','all','any','isinstance','type','map','filter','pow','Exception','ValueError']
FORBIDDEN={'eval','exec','open','compile','__import__','globals','locals','getattr','setattr','delattr','input','vars','dir','help','breakpoint','print','exit','quit'}
def admitted(code):
 try:tree=ast.parse(code)
 except SyntaxError:return False
 for n in ast.walk(tree):
  if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal)):return False
  if isinstance(n,ast.Attribute) and (n.attr.startswith('_') or n.attr in {'mro','format_map'}):return False
  if isinstance(n,ast.Name) and (n.id in FORBIDDEN or n.id.startswith('__')):return False
 return True

def execute_one():
 value=json.load(sys.stdin);code=value['code']+'\n'+'\n'.join(value['test_list'])
 if value.get('test_imports') or not admitted(code):raise ValueError('not narrowly admitted reference')
 resource.setrlimit(resource.RLIMIT_CPU,(2,2));resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20));resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
 scope={'__builtins__':{k:getattr(builtins,k) for k in SAFE}}
 exec(compile(code,'<frozen-mbpp-reference>','exec'),scope,scope)
 print('REFERENCE_TESTS_PASSED')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-one',action='store_true');p.add_argument('--registry',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--output',type=Path);p.add_argument('--count-per-class',type=int,default=16);a=p.parse_args()
 if a.execute_one:execute_one();return
 if not all([a.registry,a.profile,a.output]) or not 16<=a.count_per_class<=64:raise ValueError('bounded external references required')
 if (a.output/'REFERENCE.json').exists():print('EXISTING_IMMUTABLE_REFERENCE_REUSED');return
 import pyarrow.parquet as pq
 from p0_source_audit_v4 import registry_for
 from p0_curation_deepseek_env_v3 import load_profile
 from small_agent.curation.store import prepare
 registry=json.loads(a.registry.read_text());src=next(x for x in registry['files'] if x['task']=='mbpp_validation')
 if file_sha(Path(src['path']))!=src['sha256']:raise ValueError('Reference changed')
 selected=[];rejected=[]
 for r in sorted(pq.read_table(src['path']).to_pylist(),key=lambda x:sha(str(x['task_id']))):
  if len(selected)>=a.count_per_class:break
  if r['test_imports'] or not admitted(r['code']+'\n'+'\n'.join(r['test_list'])):rejected.append(dict(task_id=r['task_id'],reason='out_of_restricted_execution_scope'));continue
  try:
   out=subprocess.run([sys.executable,str(Path(__file__)),'--execute-one'],input=json.dumps(r),text=True,capture_output=True,timeout=5,env={'PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':''})
  except subprocess.TimeoutExpired:rejected.append(dict(task_id=r['task_id'],reason='reference_test_timeout'));continue
  if out.returncode or out.stdout.strip()!='REFERENCE_TESTS_PASSED':rejected.append(dict(task_id=r['task_id'],reason='reference_tests_failed'));continue
  selected.append(r)
 if len(selected)!=a.count_per_class:raise ValueError('Insufficient independently test-verified references')
 a.output.mkdir(parents=True,exist_ok=True);inputs=[];labels=[];evidence=[]
 for r in selected:
  text='# '+r['prompt']+'\n'+r['code']+'\n# Independently specified example tests\n'+'\n'.join(r['test_list'])+'\n'
  # Keep original code/test spelling. Verify code+tests, not an LLM-generated solution.
  for kind,body,expected in [('external_test_verified_positive',text,'keep'),('controlled_unclosed_string_negative',text+'\nBROKEN_PAYLOAD = "unterminated\n','drop')]:
   if expected=='drop':
    try:ast.parse(body)
    except SyntaxError:pass
    else:raise ValueError('Negative reference not actually syntax-broken')
   d=dict(document_id=sha(body),document_text=body,source_id='mbpp_independent_python_reference_v5',source_locator='mbpp:'+str(r['task_id'])+':'+kind,
     source_revision=src['revision'],family_id=sha('mbpp:'+str(r['task_id'])),language='python',content_kind='code_file_or_fragment',token_count=len(body.encode()),
     tokenizer_sha256=None,current_bucket='code_python',license_status='evaluation_reference_only',sampling_stratum=kind)
   inputs.append(d);labels.append(dict(document_id=d['document_id'],expected=expected,reference_kind=kind,task_id=r['task_id'],reference_basis='external authored code passes supplied tests' if expected=='keep' else 'controlled syntax corruption; NOT natural-corpus factual label'))
  evidence.append(dict(task_id=r['task_id'],code_sha256=sha(r['code']),test_sha256=sha('\n'.join(r['test_list'])),tests_passed=True,verification='restricted import-free AST admission + isolated CPU subprocess',tests=len(r['test_list'])))
 profile,_,_=load_profile(a.profile);judges=registry_for(profile)
 # Exact same judge/prompt as the source audit, for a blind calibration comparison.
 prompt=ROOT/'configs/curation/p0_source_audit_prompt_v4_r2.txt';schema=ROOT/'schemas/p0_teacher_quality_v3.json';policy=ROOT/'configs/curation/p0_source_audit_policy_parallel_v5.json'
 (a.output/'input.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in inputs));write(a.output/'labels.json',labels);write(a.output/'verification.json',dict(positive_verification=evidence,rejected_reference_candidates=rejected));write(a.output/'judges.json',judges)
 prepare(a.output/'input.jsonl',a.output/'batch',policy,schema,prompt,a.output/'judges.json')
 write(a.output/'REFERENCE.json',dict(version=5,created_unix=int(time.time()),reference_type='external_test_verified_python_and_controlled_format_negatives',training_eligible=False,
   source_file=src,documents=len(inputs),positive_documents=len(selected),negative_documents=len(selected),
   independently_verified_positive_code_tests=True,natural_corpus_factual_error_rate_certified=False,
   evidence_files=[dict(path=str(p.resolve()),sha256=file_sha(p)) for p in [a.registry,a.output/'input.jsonl',a.output/'labels.json',a.output/'verification.json',a.output/'batch/MANIFEST.json']],
   checker_sha256=file_sha(Path(__file__)),limits=['Provider may have seen MBPP in its own training; this does not prove factual generalization.','Negative cases are controlled format defects, not representative natural-document labels.','This reference can calibrate Python only; unknown languages/domains require their own independent references.']))
 print(json.dumps(dict(stage='external_reference_prepared',documents=len(inputs),paid_api_calls=0)))
if __name__=='__main__':main()
