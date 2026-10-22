import unittest

from simulator.disk import Disk
from simulator.workload import sequential_workload


class SequentialWorkloadTests(unittest.TestCase):
    def test_wraps_at_configured_disk_size(self):
        self.assertEqual(
            sequential_workload(5, start=3, disk_size=4),
            [3, 0, 1, 2, 3],
        )

    def test_rejects_start_outside_disk(self):
        with self.assertRaises(ValueError):
            sequential_workload(1, start=4, disk_size=4)

    def test_reset_returns_to_initial_head_on_small_disk(self):
        disk = Disk(size=10, initial_head=5)
        disk.move_head(7)

        disk.reset()

        self.assertEqual(disk.head_position, 5)
        self.assertEqual(disk.total_head_movement, 0)


if __name__ == "__main__":
    unittest.main()
