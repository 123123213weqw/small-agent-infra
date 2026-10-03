"""Explicit deterministic evidence context; NEVER rewrites/repairs returned labels.

The base body/protocol remain frozen. A judge generation extension binds the
adapter file hash and bounds; receipts also bind the ACTUAL network request SHA.
Original schema and exact-unique-span validation are unchanged.
"""
import json
from pathlib import Path
from .core import canonical,sha
from .supply_control import file_sha
TAG='literal_context_catalog_v1'

def catalog(text, maximum=384):
    candidates=[];seen=set()
    def add(s):
        if 8<=len(s)<=100 and s not in seen and text.count(s)==1:
            seen.add(s);candidates.append(s)
    # Full short lines or long-line windows, distributed across the complete body.
    lines=text.splitlines(keepends=True)
    if lines:
        step=max(1,len(lines)//(maximum//2))
        for line in lines[::step]:
            add(line[:100]);add(line[-100:])
    if text:
        for j in range(maximum):
            pos=(max(0,len(text)-96)*j)//max(1,maximum-1);add(text[pos:pos+96])
    if not candidates and len(text)>=8:add(text[:100])
    if len(candidates)>maximum:
        candidates=[candidates[(len(candidates)-1)*i//(maximum-1)] for i in range(maximum)]
    return candidates

def extension():
    return dict(kind=TAG,adapter_sha256=file_sha(Path(__file__)),maximum_entries=384,maximum_characters=100)

def adapt(body,doc,judge):
    cfg=judge['generation'].get('literal_context_adapter')
    if cfg is None:return body
    if cfg!=extension():raise ValueError('Frozen literal context adapter differs')
    if judge['endpoint']['kind']!='responses':raise ValueError('Responses literal context required')
    quotes=catalog(doc['document_text'],cfg['maximum_entries'])
    if not quotes:raise ValueError('No exact unique literal context; hold without API')
    result=json.loads(canonical(body));payload=json.loads(doc['review_payload'])
    payload['literal_evidence_catalog']=quotes
    result['input'][0]['content']=canonical(payload)
    return result
