"""Generic narrow independent controls and source/per-document evidence gates; not fact certification."""
from pathlib import Path
from collections import Counter,defaultdict
import ast,json,time
from datetime import datetime,timezone
from .release_gates_v5 import sha,write,bound_json
from .supply_control import file_sha,digest
from .core import quality_pass
from .store_v8 import Batch,prepare
from .evidence_ids_v8 import extension

CODE={
'python':('def add_{i}(a, b):\n    return a + b\n','\nunfinished = "'),
'javascript':('function add_{i}(a, b) {{ return a + b; }}\n','\nconst broken = "'),
 'typescript':('function add_{i}(a: number, b: number): number {{ return a + b; }}\n','\nconst broken = "'),
 'tsx':('export const value_{i}: number = {i};\n','\nconst broken = "'),
'c':('int add_{i}(int a, int b) {{ return a + b; }}\n','\nchar *broken = "'),
'cpp':('int add_{i}(int a, int b) {{ return a + b; }}\n','\nchar *broken = "'),
'csharp':('class Example{i} {{ public static int Add(int a, int b) {{ return a + b; }} }}\n','\nclass Broken {{ string value = "'),
'java':('class Example{i} {{ static int add(int a, int b) {{ return a + b; }} }}\n','\nclass Broken {{ String x = "'),
'go':('package example\nfunc add{i}(a int, b int) int {{ return a + b }}\n','\nvar broken = "'),
'rust':('fn add_{i}(a: i32, b: i32) -> i32 {{ a + b }}\n','\nlet broken = "'),
'bash':('value_{i}={i}\nprintf "%s\\n" "$value_{i}"\n','\nvalue="'),
'ruby':('def add_{i}(a, b)\n  a + b\nend\n','\nvalue = "'),
'php':('<?php function add_{i}($a, $b) {{ return $a + $b; }}\n','\n$x = "'),
'json':('{{"record": {i}, "enabled": true, "items": [1, 2, 3]}}\n','\n{"unfinished":'),
'sql':('CREATE TABLE items_{i} (id INTEGER, value TEXT);\nSELECT id FROM items_{i} WHERE id > 0;\n','\nSELECT "'),
'html':('<!DOCTYPE html><html><body><p>Example record {i}</p></body></html>\n','\n<broken attribute="'),
'css':('.example_{i} {{ color: red; margin: {i}px; }}\n','\n.broken {{ value: "'),
'lua':('local function add_{i}(a, b) return a + b end\n','\nlocal x = "'),
'kotlin':('fun add{i}(a: Int, b: Int): Int = a + b\n','\nval x = "'),
'swift':('func add{i}(_ a: Int, _ b: Int) -> Int {{ return a + b }}\n','\nlet x = "'),
'perl':('sub add_{i} {{ my ($a, $b) = @_; return $a + $b; }}\n','\nmy $x = "'),
'r':('add_{i} <- function(a, b) {{ a + b }}\n','\nx <- "'),
'scala':('object Example{i} {{ def add(a: Int, b: Int): Int = a + b }}\n','\nval x = "'),
'haskell':('add{i} a b = a + b\n','\nx = "'),
}

def control_rows(domain,language):
    rows=[];labels=[];verification=[]
    for i in range(1,17):
        if domain=='code':
            from tree_sitter_language_pack import get_parser
            if language not in CODE:raise ValueError('independent_language_reference_missing:'+language)
            parser=get_parser(language);fmt,broken=CODE[language];positive=fmt.format(i=i)
            if parser.parse(positive.encode()).root_node.has_error:raise ValueError('positive_parser_control_invalid:'+language)
            negative=positive+broken.replace('{{','{')
            if not parser.parse(negative.encode()).root_node.has_error:raise ValueError('negative_parser_control_not_broken:'+language)
            if language=='python':ast.parse(positive)
            # A brief language-native comment makes full short code evidence addressable.
            basis='authored_complete_code_syntax_vs_unterminated_string; parser-only, not runtime correctness'
        else:
            payload={'record':i,'items':[i,i+1,i+2],'count':3};encoded=json.dumps(payload,ensure_ascii=False)
            if json.loads(encoded)['count']!=len(payload['items']):raise ValueError('independent JSON/count control invalid')
            positive=(f'记录 {i}：读取与检查结构化数据\n\n下面是一个完整 JSON 对象：\n{encoded}\n\nrecord 是整数编号，items 是三个整数组成的数组，count 表示数组长度。解析后可以先检查 items 是数组，再比较它的长度与 count；本例长度为 3，两者一致。true、false 和 null 是 JSON 的字面量，字符串键需要用双引号包围。将数据与解析逻辑分开保存，可以在不修改程序的情况下更换记录。读取失败时应报告错误，而不是把残缺文本当成完整对象。\n' if language=='zh' else f'Record {i}: reading and checking structured data\n\nThe following is a complete JSON object:\n{encoded}\n\nThe record field is an integer identifier. The items field is an array of three integers, and count records its length. After parsing, check that items is an array and compare its length with count. In this example the length is 3, so the check succeeds. JSON object keys use double quotes; booleans and null are literals rather than quoted strings. Keeping data separate from the parsing logic lets a program process a different record without changing the program. A parse failure should be reported rather than treating an unfinished object as a complete record.\n')
            negative=(f'入口 {i}。立即订阅！点击广告购买答案。注册登录后才能查看完整正文。这里没有任何正文，只有付费预览和广告链接。立即订阅！点击广告购买答案。\n' if language=='zh' else f'Portal {i}. Subscribe now! Click the sponsored links. Register and pay to unlock the full article and answer. There is no article body here, only advertising links and a paid preview. Subscribe now! Click the sponsored links.\n')
            basis='authored_JSON_syntax_and_array_length_positive_vs_explicit_bodyless_advertising_negative; narrow_integrity_only'
        for text,expected in ((positive,'keep'),(negative,'drop')):
            h=sha(text);rows.append(dict(document_id=h,text=text,normalized_sha256=h,source_id='independent_controls_v11_'+domain+'_'+language,source_locator='control:'+h,source_revision='v11',family_id=sha(str(i)+domain+language),language=language,domain=domain,own_text_tokens=max(1,len(text.encode())),split='test'))
            labels.append(dict(document_id=h,expected=expected,independent_basis=basis));verification.append(dict(document_id=h,check=basis,passed=True))
    return rows,labels,verification

def make_batch(cfg,folder,rows,meta):
    folder=Path(folder)
    if (folder/'PLAN.json').exists():
        plan=bound_json(folder/'PLAN.json')
        if plan['identity']!=digest(meta):raise ValueError('Frozen review cohort differs')
        b=Batch(folder/'batch');b.close();return plan
    import shutil,tempfile,sys
    from p0_source_audit_v4 import registry_for
    from p0_curation_deepseek_env_v3 import load_profile
    profile,_,_=load_profile(Path(cfg['profile']));registry=registry_for(profile);registry['judges']['deepseek']['generation']['evidence_id_transport']=extension()
    schema=json.loads(Path(cfg['schema']).read_text());schema['properties']['findings']['items']['properties']['quotes']['maxItems']=8
    folder.parent.mkdir(parents=True,exist_ok=True);tmp=Path(tempfile.mkdtemp(prefix='.review-',dir=folder.parent))
    try:
        write(tmp/'schema.json',schema);write(tmp/'judges.json',registry);inputs=[]
        for r in rows:
            inputs.append(dict(document_id=r['normalized_sha256'],document_text=r['text'],source_id=r['source_id'],source_locator=r['source_locator'],source_revision=r['source_revision'],family_id=r['family_id'],language=r['language'],content_kind=('complete_code_file' if meta['kind']=='reference' else 'code_file_or_fragment') if r['domain']=='code' else 'text',token_count=r['own_text_tokens'],tokenizer_sha256=cfg['tokenizer_sha256'],license_status='evaluation_only' if meta['kind']=='reference' else 'internal_source_scope_pending_release',sampling_stratum=meta.get('strata',{}).get(r['normalized_sha256'],'individual')))
        (tmp/'input.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in inputs))
        prepare(tmp/'input.jsonl',tmp/'batch',Path(cfg['audit_policy']),tmp/'schema.json',Path(cfg['prompt']),tmp/'judges.json')
        plan=dict(version=11,identity=digest(meta),meta=meta,documents=len(inputs),body_ids=sorted(r['normalized_sha256'] for r in rows),evidence_files=[dict(path=str((folder/n).resolve()),sha256=file_sha(tmp/n)) for n in ('input.jsonl','schema.json','judges.json')])
        write(tmp/'PLAN.json',plan);tmp.rename(folder);return plan
    except BaseException:shutil.rmtree(tmp,ignore_errors=True);raise

def annotations(folder):
    folder=Path(folder)
    plan=bound_json(folder/('PLAN.json' if (folder/'PLAN.json').exists() else 'REFERENCE.json'));b=Batch(folder/'batch')
    try:
        results={r[0]:json.loads(r[1]) for r in b.db.execute("SELECT document_id,annotation FROM jobs WHERE status='valid'")}
        approved={h for h,a in results.items() if quality_pass(a,b.policy) and not any(x['severity']=='hold' for x in b.docs[h]['lint_flags'])}
        return results,approved,b.state_fingerprint(),b.manifest_hash
    finally:b.close()

def reference_gate(folder,labels,policy):
    folder=Path(folder)
    if (folder/'REFERENCE_V11.json').exists():
        ref=bound_json(folder/'REFERENCE_V11.json')
        if ref['checker_sha256']!=file_sha(Path(__file__)) or ref['training_eligible'] is not False:raise ValueError('Independent reference checker changed')
        _,expected,_=control_rows(ref['domain'],ref['language'])
        if labels!=expected:raise ValueError('Independent control labels differ')
    else:
        ref=bound_json(folder/'REFERENCE.json')
        if ref.get('training_eligible') is not False or not ref.get('independently_verified_arithmetic_steps'):raise ValueError('Bound independently checked math reference required')
    b=Batch(folder/'batch')
    try:
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(b.manifest['created_at'])).total_seconds()/86400
        if age<0 or age>policy.get('maximum_reference_age_days',30):raise ValueError('Independent reference expired or clock invalid')
    finally:b.close()
    ann,approved,state,manifest=annotations(folder);c=Counter();holds=[]
    for row in labels:
        k=row['expected'];c[k+'_total']+=1;c[k+'_valid']+=row['document_id'] in ann;c[k+'_keep']+=row['document_id'] in approved
    for k in ('keep','drop'):
        if c[k+'_total']<policy['minimum_reference_per_class'] or c[k+'_valid']/max(1,c[k+'_total'])<policy['minimum_valid_rate']:holds.append('reference_coverage_'+k)
    if c['drop_keep']>policy['maximum_reference_false_keep']:holds.append('controlled_false_keep')
    if c['keep_keep']/max(1,c['keep_total'])<policy['minimum_reference_keep_rate']:holds.append('positive_keep_rate')
    return dict(status='hold' if holds else 'passed',holds=holds,counts=dict(c),review_state_sha256=state,batch_manifest_sha256=manifest,training_eligible=False,natural_corpus_factual_error_rate_certified=False)

def source_gate(folder,policy):
    plan=bound_json(Path(folder)/'PLAN.json');ann,approved,state,manifest=annotations(folder);counts=defaultdict(Counter)
    for h,label in plan['meta']['strata'].items():
        c=counts[label];c['selected']+=1;c['valid']+=h in ann;c['keep']+=h in approved
    passed=[k for k,c in counts.items() if c['selected']>=policy['minimum_source_samples_per_stratum'] and c['valid']/c['selected']>=policy['minimum_valid_rate'] and c['keep']/c['selected']>=policy['minimum_joint_keep_rate']]
    return dict(status='passed' if passed else 'hold',approved_strata=sorted(passed),held_strata=sorted(set(counts)-set(passed)),counts={k:dict(v) for k,v in counts.items()},known_bad_ids=sorted(set(plan['body_ids'])-approved),review_state_sha256=state,batch_manifest_sha256=manifest)

def publication_quality(cfg,candidates,source_folder,reference_folder,batch_folder,mode):
    """Re-read exact teacher states. Failed source/batch strata require ALL retained rows individually reviewed."""
    import pyarrow.parquet as pq
    policy=json.loads(Path(cfg['release_policy']).read_text())['quality'];ref=Path(reference_folder)
    gate=reference_gate(ref,json.loads((ref/'labels.json').read_text()),policy)
    if gate['status']!='passed':raise ValueError('Independent reference failed')
    sg=source_gate(source_folder,policy);plan=bound_json(Path(batch_folder)/'PLAN.json')
    if plan['meta']['candidate_sha256']!=file_sha(Path(candidates)) or plan['meta']['mode']!=mode:raise ValueError('Candidate review binding differs')
    ann,approved,state,manifest=annotations(batch_folder);known=set(sg['known_bad_ids']);keep=set();reasons=Counter()
    rows=pq.read_table(candidates).to_pylist()
    source_plan=bound_json(Path(source_folder)/'PLAN.json')
    meta=source_plan['meta']
    if plan['meta'].get('source_manifest_sha256')!=file_sha(Path(source_folder)/'PLAN.json'):raise ValueError('Source calibration binding differs')
    if not rows or any(r['source_id']!=meta['source_id'] or r['domain']!=meta['domain'] or r['language']!=meta['language'] for r in rows):raise ValueError('Source/domain/language calibration mismatch')
    if (ref/'REFERENCE_V11.json').exists():
        ref_binding=bound_json(ref/'REFERENCE_V11.json')
        if (ref_binding['domain'],ref_binding['language'])!=(meta['domain'],meta['language']):raise ValueError('Wrong independent reference domain/language')
    elif (meta['domain'],meta['language'])!=('math','en'):raise ValueError('Arithmetic reference cannot calibrate another domain')
    if mode=='individual':
        if set(plan['body_ids'])!={r['normalized_sha256'] for r in rows}:raise ValueError('Every document must be in bound individual review')
        keep=approved-known
    elif mode=='source_audit':
        counts=defaultdict(Counter)
        for h,s in plan['meta']['strata'].items():
            c=counts[s];c['selected']+=1;c['valid']+=h in ann;c['keep']+=h in approved
        strata={s for s,c in counts.items() if c['selected']>=min(8,plan['meta']['population'][s]) and c['valid']/c['selected']>=policy['minimum_valid_rate'] and c['keep']/c['selected']>=policy['minimum_joint_keep_rate']}
        strata &= set(sg['approved_strata']);known |= set(plan['body_ids'])-approved
        from .full_inventory_v11 import layer
        keep={r['normalized_sha256'] for r in rows if layer(r) in strata and r['normalized_sha256'] not in known}
    else:raise ValueError('Unknown release mode')
    evidence=[Path(source_folder)/'PLAN.json',Path(batch_folder)/'PLAN.json',ref/('REFERENCE_V11.json' if (ref/'REFERENCE_V11.json').exists() else 'REFERENCE.json')]
    return dict(status='passed' if keep else 'hold',source_id=meta['source_id'],kept_ids=sorted(keep),excluded_ids=sorted({r['normalized_sha256'] for r in rows}-keep),mode=mode,source_manifest_sha256=sg['batch_manifest_sha256'],reference_gate=gate,source_gate=sg,batch_state_sha256=state,batch_manifest_sha256=manifest,natural_corpus_factual_error_rate_certified=False,individual_dual_review=False,evidence_files=[dict(path=str(f.resolve()),sha256=file_sha(f)) for f in evidence])
