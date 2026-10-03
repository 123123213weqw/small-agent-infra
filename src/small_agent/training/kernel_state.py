"""Checkpoint Triton 3.1 autotune *choices*, not compiled binaries or code.

Different timing winners can change floating-point reduction order without
consuming RNG. Restoring only random states is therefore not bitwise recovery.
Only declared, locally installed kernel configs may be restored.
"""
import hashlib
from pathlib import Path
import sys


def registry():
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return {}
    found={}
    for name,module in sorted(list(sys.modules.items())):
        if not name.startswith(('fla.','liger_kernel.')) or module is None:continue
        for obj in list(vars(module).values()):
            seen=set()
            while obj is not None and id(obj) not in seen:
                seen.add(id(obj))
                if isinstance(obj,Autotuner):
                    base=obj.base_fn
                    found[f'{base.__module__}:{base.__name__}']=obj
                    break
                # Known Triton decorator chains only, never arbitrary objects.
                if type(obj).__module__.startswith('triton.'):
                    obj=getattr(obj,'fn',None)
                else:break
    return found


def describe(config):
    return {key:getattr(config,key) for key in ('kwargs','num_warps','num_stages','num_ctas','maxnreg')}


def kernel_digest(tuner):
    return hashlib.sha256(Path(tuner.base_fn.__code__.co_filename).read_bytes()).hexdigest()


def capture_kernel_state():
    try:import triton
    except ImportError:return None
    entries={}
    for name,tuner in registry().items():
        if tuner.cache:
            entries[name]={'source_sha256':kernel_digest(tuner),
                'choices':[{'key':list(key),'config':describe(config)} for key,config in tuner.cache.items()]}
    return {'version':'triton_autotune_choices_v1','triton':triton.__version__,'entries':entries}


def restore_kernel_state(state):
    if state is None:return
    import triton
    if state.get('version')!='triton_autotune_choices_v1' or state['triton']!=triton.__version__:
        raise ValueError('Triton autotune checkpoint runtime changed')
    installed=registry()
    # Validate everything before mutating any local runtime cache.
    plan=[]
    for name,entry in state['entries'].items():
        if name not in installed:raise ValueError('checkpoint kernel not installed: '+name)
        tuner=installed[name]
        if kernel_digest(tuner)!=entry['source_sha256']:raise ValueError('checkpoint kernel source changed: '+name)
        for choice in entry['choices']:
            matching=[config for config in tuner.configs if describe(config)==choice['config']]
            if not matching:raise ValueError('checkpoint kernel config is not a declared installed choice')
            if not all(isinstance(x,(str,int,float,bool,type(None))) for x in choice['key']):raise ValueError('nonliteral autotune key')
            plan.append((tuner,tuple(choice['key']),matching[0]))
    for tuner,key,config in plan:tuner.cache[key]=config
