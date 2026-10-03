import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT/'scripts/benchmark_perf8_gpu_guard_v1.py'
if not SCRIPT.exists():SCRIPT=ROOT/'benchmark_perf8_gpu_guard_v1.py'
spec=importlib.util.spec_from_file_location('gpu_guard',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

GROUP='0::/user.slice/app.slice/p0-wave2-perf8-v1-retry1.service\n'
class GuardTest(unittest.TestCase):
    def test_own_parent_allowed(self):m.validate_parent(12,12,GROUP,GROUP)
    def test_wrong_parent_rejected(self):
        with self.assertRaises(ValueError):m.validate_parent(13,12,GROUP,GROUP)
    def test_root_parent_rejected(self):
        with self.assertRaises(ValueError):m.validate_parent(1,1,GROUP,GROUP)
    def test_other_cgroup_rejected(self):
        with self.assertRaises(ValueError):m.validate_parent(12,12,GROUP,'other')
    def test_nonbenchmark_cgroup_rejected(self):
        with self.assertRaises(ValueError):m.validate_parent(12,12,'generic','generic')
    def test_identifies_external_not_own_process(self):
        ps=[{'pid':1},{'pid':2}]
        result=m.outside_processes(ps,GROUP,lambda p:GROUP if p==1 else 'other')
        self.assertEqual([p['pid'] for p in result],[2])
    def test_exited_process_ignored(self):
        def gone(p):raise FileNotFoundError()
        self.assertEqual(m.outside_processes([{'pid':1}],GROUP,gone),[])
    def test_unreadable_cgroup_does_not_silently_pass(self):
        def no_permission(p):raise PermissionError()
        with self.assertRaises(PermissionError):m.outside_processes([{'pid':1}],GROUP,no_permission)

if __name__=='__main__':unittest.main()
