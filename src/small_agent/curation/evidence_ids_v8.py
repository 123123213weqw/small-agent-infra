"""Versioned evidence-ID wire protocol; lossless references, never decision repair."""
import copy,json
from pathlib import Path
from jsonschema import Draft202012Validator
from .core import canonical,fingerprint,validate_annotation,span
from .supply_control import file_sha
from .literal_context_v6 import catalog

def extension():
    return dict(kind='evidence_ids_v2_conservative_veto',adapter_sha256=file_sha(Path(__file__)),catalog_sha256=file_sha(Path(__file__).with_name('literal_context_v6.py')),maximum_entries=384)

def spans(doc):
    text=doc['document_text'];items=[]
    for i,q in enumerate(catalog(text)):
        try:location=span(text,q)
        except ValueError:continue # .count misses OVERLAPPING repeated occurrences.
        start=location['start'];items.append(dict(id='e'+str(i).zfill(3),start=start,end=start+len(q),text=q))
    if not items:raise ValueError('No uniquely grounded evidence; hold without API')
    return items

def check_judge(judge):
    if judge['generation'].get('evidence_id_transport')!=extension() or judge['endpoint']['kind']!='responses':raise ValueError('Frozen evidence ID transport differs')

def wire_schema(doc,schema,judge):
    check_judge(judge);s=copy.deepcopy(schema);ids=[x['id'] for x in spans(doc)];p=s['properties']
    for key in ('document_id','body_sha256'):p[key]={'type':'string','const':doc[key]}
    p['anchors']['items']={'type':'string','enum':ids};p['anchors']['maxItems']=1
    p['findings']['items']['properties']['quotes']['items']={'type':'string','enum':ids}
    p['findings']['maxItems']=3
    return s

def request_body(doc,prompt,schema,judge):
    check_judge(judge);payload=json.loads(doc['review_payload']);payload['evidence_spans']=spans(doc)
    g=judge['generation']
    return dict(model=judge['model'],instructions=prompt,input=[dict(role='user',content=canonical(payload))],reasoning=dict(effort='none'),temperature=g['temperature'],max_output_tokens=g['max_tokens'],stream=False,tool_choice='none',text=dict(format=dict(type='json_schema',name='quality_evidence_ids_v8',schema=wire_schema(doc,schema,judge))))

def resolve(raw,doc,schema,judge):
    Draft202012Validator(wire_schema(doc,schema,judge)).validate(raw)
    mapping={x['id']:x['text'] for x in spans(doc)};out=copy.deepcopy(raw)
    out['anchors']=[mapping[x] for x in raw['anchors']]
    for f in out['findings']:f['quotes']=[mapping[x] for x in f['quotes']]
    # Fail closed: contradictory KEEP can only become HOLD, never approval.
    if out['decision']=='keep' and any(f['severity'] in ('review','reject') for f in out['findings']):out['decision']='uncertain'
    validate_annotation(out,doc,schema) # Original full-body uniqueness remains mandatory.
    return out

def validate_receipt(result,doc,prompt,schema,judge):
    body=request_body(doc,prompt,schema,judge)
    if result.get('actual_request_sha256')!=fingerprint(body):raise ValueError('Actual request binding differs')
    normalized=resolve(result['wire_annotation'],doc,schema,judge)
    if normalized!=result.get('annotation') or normalized!=result.get('raw_annotation'):raise ValueError('Evidence resolver/decision was altered')
    return normalized
