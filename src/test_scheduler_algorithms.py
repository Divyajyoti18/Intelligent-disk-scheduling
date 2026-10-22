import unittest

from schedulers.algorithms import cscan, scan


class EmptyQueueSchedulingTests(unittest.TestCase):
    def test_scan_does_not_move_for_empty_queue(self):
        self.assertEqual(scan([], 53), ([], 0))
        self.assertEqual(scan([], 53, direction="left"), ([], 0))

    def test_cscan_does_not_wrap_for_empty_queue(self):
        self.assertEqual(cscan([], 53), ([], 0))


if __name__ == "__main__":
    unittest.main()
