#!/usr/bin/env python3
"""One explicit protocol correction on the SAME immutable cohort, not resampling."""
import argparse,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from small_agent.curation.evidence_ids_v7 import extension
from small_agent.curation.store_v7 import prepare
from small_agent.curation.supply_control import file_sha
from small_agent.curation.release_gates_v5 import write

def rebind(source,output,prompt,kind):
    if (output/'PROTOCOL_REBIND.json').exists():
        r=json.loads((output/'PROTOCOL_REBIND.json').read_text())
        if r['source_input_sha256']!=file_sha(source/'input.jsonl') or r['prompt_sha256']!=file_sha(prompt) or r['adapter']!=extension():raise ValueError('Immutable rebind differs')
        return
    output.mkdir(parents=True,exist_ok=True)
    registry=json.loads((source/'batch/judges.json').read_text());registry['judges']['deepseek']['generation'].pop('literal_context_adapter',None)
    registry['judges']['deepseek']['generation']['evidence_id_transport']=extension()
    for n in ['input.jsonl','selection.jsonl','labels.json','verification.json']:
        if (source/n).exists():shutil.copyfile(source/n,output/n)
    write(output/'judges.json',registry)
    prepare(output/'input.jsonl',output/'batch',source/'batch/policy.json',source/'batch/schema.json',prompt,output/'judges.json')
    if kind=='source':
        plan=json.loads((source/'SAMPLE_PLAN.json').read_text());plan['protocol_correction_of']=str(source.resolve())
        # Original sampled rows, denominator, source-file identities are retained.
        for n in plan['frozen_files']:
            if not (output/n).exists():shutil.copyfile(source/n,output/n)
            plan['frozen_files'][n]=file_sha(output/n)
        write(output/'SAMPLE_PLAN.json',plan)
    else:
        ref=json.loads((source/'REFERENCE.json').read_text());ref.update(created_unix=int(time.time()),protocol_correction_of=str(source.resolve()))
        ref['evidence_files']=[dict(path=str((output/Path(e['path']).name).resolve()),sha256=file_sha(output/Path(e['path']).name)) if (output/Path(e['path']).name).exists() else e for e in ref['evidence_files']]
        # Manifest lives inside batch, not at output root.
        ref['evidence_files']=[dict(path=str((output/'batch/MANIFEST.json').resolve()),sha256=file_sha(output/'batch/MANIFEST.json')) if Path(e['path']).name=='MANIFEST.json' else e for e in ref['evidence_files']]
        write(output/'REFERENCE.json',ref)
    write(output/'PROTOCOL_REBIND.json',dict(version=6,reason='Replace fragile literal generation with evidence ID transport; exact same cohort, gates unchanged',source=str(source.resolve()),source_input_sha256=file_sha(source/'input.jsonl'),prompt_sha256=file_sha(prompt),adapter=extension(),same_cohort=True,resampled=False))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['source','output','prompt']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--kind',choices=['source','reference'],required=True);a=p.parse_args();rebind(a.source,a.output,a.prompt,a.kind)
if __name__=='__main__':main()
