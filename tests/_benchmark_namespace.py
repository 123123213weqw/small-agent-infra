"""CLI benchmark wrappers configure their engine at import; isolate test namespaces."""
import importlib
from pathlib import Path
import sys


def import_benchmarks(*names):
    scripts=str(Path(__file__).parents[1]/'scripts')
    if scripts not in sys.path:sys.path.insert(0,scripts)
    keys=['benchmark_microbatch_v1','benchmark_perf8_v1','benchmark_loss_backend_v1',
          'benchmark_loss8_v1','benchmark_loss8_gpu_guard_v1','benchmark_fusion_backend_v1',
          'benchmark_fusion8_v1','benchmark_fusion8_gpu_guard_v1',
          'benchmark_fusion4_confirm_v1','benchmark_fusion4_confirm_gpu_guard_v1',
          'benchmark_fusion8_confirm_v1','benchmark_fusion8_confirm_gpu_guard_v1',
          'benchmark_fusion8_tune_v1','benchmark_fusion8_tune_gpu_guard_v1',
          'benchmark_fusion8_tune_retry_v1','benchmark_fusion8_tune_retry_gpu_guard_v1']
    saved={key:sys.modules.get(key) for key in keys}
    for key in keys:sys.modules.pop(key,None)
    try:return tuple(importlib.import_module(name) for name in names)
    finally:
        for key,value in saved.items():
            if value is None:sys.modules.pop(key,None)
            else:sys.modules[key]=value
