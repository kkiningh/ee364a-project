import unittest
import numpy as np

from accel import accel

class MainTest(unittest.TestCase):
    def test_run(self):
        runner = accel.KernelRunner(10)

        A = np.random.randn(4, 4).astype(np.float32)
        B = np.random.randn(4, 2).astype(np.float32)
        w = runner.run(A, B, 10)

if __name__ == '__main__':
    unittest.main()
