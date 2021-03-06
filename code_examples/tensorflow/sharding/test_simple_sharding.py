# Copyright 2019 Graphcore Ltd.
import inspect
import numpy as np
import os
import subprocess
import sys
import unittest

import tests.test_util as tu


def run_simple_sharding(autoshard):
    cwd = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))    
    py_version = "python{}".format(sys.version_info[0])
    cmd = [py_version, "simple_sharding.py"]
    if autoshard:
        cmd.append("--autoshard")
    out = subprocess.check_output(cmd, cwd=cwd, universal_newlines=True)
    return out


class TestTensorFlowSharding(unittest.TestCase):
    """High-level integration tests for tensorflow sharding examples"""

    def test_manual_sharding(self):
        """Manual sharding example using 2 shards"""
        out = run_simple_sharding(False)
        tu.assert_result_equals_tensor_value(
            out, np.array([3.0, 8.0], dtype=np.float32)
        )

    def test_auto_sharding(self):
        """Automatic sharding example using 2 shards"""
        out = run_simple_sharding(True)
        tu.assert_result_equals_tensor_value(
            out, np.array([3.0, 8.0], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
