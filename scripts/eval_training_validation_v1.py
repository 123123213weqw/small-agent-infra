#!/usr/bin/env python3
"""Read-only full/quick domain evaluation; torchrun optional, no DDP training."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
from training_eval_common_v1 import fixed_subset, sha, verify_inputs, weighted_loss, write_json


def main():
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset
    from safetensors.torch import load_model
    from small_agent.data import ShardedTokenDataset
    from small_agent.evaluation import evaluate_causal_lm
    from small_agent.models import build_model, load_model_spec
    from torch.utils.tensorboard import SummaryWriter
    p = argparse.ArgumentParser()
    for k in ('training_run','checkpoint','tokenizer_dir','config','output'):
        p.add_argument('--'+k.replace('_','-'), type=Path, required=True)
    p.add_argument('--scope', choices=['quick','full'], default='full')
    a = p.parse_args()
    rank, world, local = int(os.getenv('RANK',0)), int(os.getenv('WORLD_SIZE',1)), int(os.getenv('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    device = torch.device('cuda',local)
    if world > 1: dist.init_process_group('nccl')
    cfg = json.loads(a.config.read_text())
    rm = json.loads((a.training_run/'run_manifest.json').read_text())
    cm = json.loads((a.checkpoint/'manifest.json').read_text())
    if rank == 0:
        _, _, identity = verify_inputs(a.training_run, a.checkpoint, a.tokenizer_dir)
        if a.output.exists(): raise FileExistsError(a.output)
        a.output.mkdir(parents=True)
    if world > 1: dist.barrier()
    t0 = time.monotonic()
    data = ShardedTokenDataset(Path(rm['data_manifest']['path']).parent, 'validation', cfg['context_length'], verify_hashes=rank==0)
    groups = defaultdict(list)
    for ref in range(len(data)):
        groups[str(data.reference(ref)['domain'])].append(ref)
    model = build_model(load_model_spec(rm['model_spec']['path']))
    missing, unexpected = load_model(model, str(a.checkpoint/'model.safetensors'), strict=True)
    if missing or unexpected: raise ValueError('model state mismatch')
    model.to(device).eval()
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    results, ref_plan = {}, {}
    for domain, refs in sorted(groups.items()):
        chosen = fixed_subset(refs, cfg['quick_max_samples_per_raw_domain'] if a.scope=='quick' else None)
        ref_plan[domain] = chosen
        loader = DataLoader(Subset(data, chosen[rank::world]), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        v = evaluate_causal_lm(model, loader, device)
        bc = torch.tensor(v.batches,device=device,dtype=torch.int64)
        if world > 1: dist.all_reduce(bc)
        results[domain] = {'loss':v.loss, 'ppl':v.perplexity, 'predicted_tokens':v.predicted_tokens, 'samples':len(chosen), 'available_samples':len(refs), 'batches':int(bc), 'coverage_warning':'small_domain' if len(chosen)<128 else None}
        if rank == 0:
            print(json.dumps({'domain':domain, **results[domain]}),flush=True)
            write_json(a.output/'progress.json',results)
    if rank == 0:
        overall = weighted_loss(results)
        write_json(a.output/'sample_refs.json',ref_plan)
        report = {'version':'training_validation_v1','scope':a.scope,'checkpoint_step':cm['step'],'tokens_seen':cm['tokens_seen'], 'identity':identity, 'config_sha256':sha(a.config),'precision':cfg['precision'],'context_length':cfg['context_length'],'sample_refs_sha256':sha(a.output/'sample_refs.json'), 'domains':results,'overall':overall,'seconds':time.monotonic()-t0,'world_size':world}
        write_json(a.output/'result.json', report)
        with SummaryWriter(str(a.output/'tensorboard')) as tb:
            for name, value in {'overall':overall,**results}.items():
                for key in ('loss','ppl','predicted_tokens'):
                    tb.add_scalar(f'eval/{a.scope}/{name}/{key}',value[key],cm['step'])
            tb.add_text('eval/protocol',json.dumps(report,ensure_ascii=False,indent=2),cm['step'])
        print(json.dumps({'complete':True,**overall}),flush=True)
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__': main()
