#!/usr/bin/env python3
"""Independent authored GSM8K arithmetic controls, evaluation-only, no corpus approval."""
import argparse, ast, json, re, sys, time
from fractions import Fraction
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.release_gates_v5 import sha,write
from small_agent.curation.supply_control import file_sha

def arithmetic(expression):
    if len(expression)>200:raise ValueError('arithmetic too large')
    tree=ast.parse(expression.replace(',',''),mode='eval')
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            if abs(n.value)>10**12:raise ValueError('large number')
            return Fraction(str(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            v=visit(n.operand);return -v if isinstance(n.op,ast.USub) else v
        if isinstance(n,ast.BinOp):
            a,b=visit(n.left),visit(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
        raise ValueError('unsupported arithmetic AST')
    return visit(tree.body)

def verified_steps(answer):
    steps=[]
    for m in re.finditer(r'<<([^<>]+)>>',answer):
        equation=m.group(1)
        if equation.count('=')!=1:raise ValueError('non-simple annotated equation')
        expression,value=equation.split('=')
        actual=arithmetic(expression);expected=arithmetic(value)
        if actual!=expected:raise ValueError('annotated arithmetic not exact')
        steps.append(dict(expression=expression,result=value,actual=str(actual)))
    if not steps:raise ValueError('no annotated arithmetic')
    return steps

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('registry','profile','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--count-per-class',type=int,default=32);a=p.parse_args()
    if not 16<=a.count_per_class<=64:raise ValueError('bounded reference required')
    if (a.output/'REFERENCE.json').exists():print('EXISTING_MATH_REFERENCE');return
    import pyarrow.parquet as pq
    from p0_source_audit_v4 import registry_for
    from p0_curation_deepseek_env_v3 import load_profile
    from small_agent.curation.store import prepare
    registry=json.loads(a.registry.read_text());src=next(f for f in registry['files'] if f['task']=='gsm8k_test')
    if file_sha(Path(src['path']))!=src['sha256']:raise ValueError('external reference changed')
    chosen=[];reject=[]
    for index,row in sorted(enumerate(pq.read_table(src['path']).to_pylist()),key=lambda t:sha(str(t[0]))):
        try:steps=verified_steps(row['answer'])
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):reject.append(index);continue
        chosen.append((index,row,steps))
        if len(chosen)==a.count_per_class:break
    if len(chosen)!=a.count_per_class:raise ValueError('insufficient external arithmetic references')
    a.output.mkdir(parents=True,exist_ok=True);docs=[];labels=[];checks=[]
    for index,row,steps in chosen:
        positive='Problem: '+row['question']+'\nWorked solution:\n'+row['answer']+'\n'
        negative=positive
        for s in steps:
            old='<<'+s['expression']+'='+s['result']+'>>'
            negative=negative.replace(old,'<<'+s['expression']+'='+str(arithmetic(s['expression'])+10000)+'>>')
        # Visible equations, not hidden annotations only. Contradictions are controlled.
        negative+='\nCalculation summary (claimed results):\n'+'\n'.join(s['expression']+' = '+str(arithmetic(s['expression'])+10000) for s in steps)+'\n'
        for kind,body,expected in [('external_authored_arithmetic_positive',positive,'keep'),('controlled_arithmetic_contradiction',negative,'drop')]:
            docs.append(dict(document_id=sha(body),document_text=body,source_id='gsm8k_arithmetic_reference_v6',
                source_locator=f'gsm8k_test:{index}:{kind}',source_revision=src['revision'],family_id=sha('gsm8k:'+str(index)),
                language='en',content_kind='text',token_count=len(body.encode()),tokenizer_sha256=None,
                license_status='evaluation_reference_only',sampling_stratum=kind,current_bucket='math_en'))
            labels.append(dict(document_id=sha(body),expected=expected,reference_kind=kind,source_row=index))
        checks.append(dict(source_row=index,verified_annotated_steps=steps,negative_offset=10000))
    profile,_,_=load_profile(a.profile)
    (a.output/'input.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in docs))
    write(a.output/'labels.json',labels);write(a.output/'verification.json',dict(checks=checks,rejected_rows=reject));write(a.output/'judges.json',registry_for(profile))
    prepare(a.output/'input.jsonl',a.output/'batch',ROOT/'configs/curation/p0_source_audit_policy_parallel_v5.json',ROOT/'schemas/p0_teacher_quality_v3.json',ROOT/'configs/curation/p0_source_audit_prompt_v4_r2.txt',a.output/'judges.json')
    write(a.output/'REFERENCE.json',dict(version=6,created_unix=int(time.time()),reference_type='external_authored_arithmetic_checks_and_controlled_contradictions',
        language='en',domain='math',training_eligible=False,independently_verified_arithmetic_steps=True,
        natural_corpus_factual_error_rate_certified=False,source_file=src,positive_documents=len(chosen),negative_documents=len(chosen),
        checker_sha256=file_sha(Path(__file__)),evidence_files=[dict(path=str(f.resolve()),sha256=file_sha(f)) for f in [a.registry,a.output/'input.jsonl',a.output/'labels.json',a.output/'verification.json',a.output/'batch/MANIFEST.json']],
        limits=['Arithmetic annotations only: not a proof of whole natural-document correctness.','Controlled contradictions are not naturally sampled error labels.','Provider may have seen this benchmark; source exact overlap and evaluation contamination must still be screened.']))
    print(json.dumps(dict(stage='math_reference_prepared',documents=len(docs),paid_api_calls=0)))
if __name__=='__main__':main()
