"""Reuse frozen v12 transport/cursor/isolation fixtures for the new coordinator."""
from unittest.mock import patch
import unittest
import test_distributed_scan_v12 as original
import p0_distributed_scan_v13 as optimized

class DistributedParallelTests(original.DistributedTests):
    def setUp(self):
        super().setUp()
        p=patch.object(original,'coord',optimized);p.start();self.addCleanup(p.stop)

if __name__=='__main__':unittest.main()
