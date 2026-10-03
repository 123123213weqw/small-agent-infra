"""Generate with official lm-eval stop criteria and an explicit audited cache policy."""
def generate(lm, context, max_length, stop, cache_ok, kwargs):
    import torch
    from lm_eval.models.utils_hf import stop_sequences_criteria
    kw=dict(kwargs)
    kw.pop('use_cache',None)
    kw.setdefault('do_sample',False)
    if kw['do_sample'] is False: kw.pop('temperature',None)
    criteria=stop_sequences_criteria(lm.tokenizer,stop,context.shape[1],context.shape[0])
    with torch.no_grad(),torch.autocast(lm.device.type,dtype=lm.mixed_precision_dtype):
        return lm.model.generate(input_ids=context,max_length=max_length,stopping_criteria=criteria,pad_token_id=lm.tokenizer.pad_token_id,use_cache=cache_ok,**kw)
