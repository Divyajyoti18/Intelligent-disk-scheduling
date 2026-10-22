import unittest

from simulator.predictive_cache import PredictiveCache


class PredictiveCacheRegressionTests(unittest.TestCase):
    def test_sequential_prediction_keeps_configured_history(self):
        cache = PredictiveCache(
            capacity=8,
            history_size=2,
            sequential_min_length=3,
            disk_size=10,
        )

        for position in [8, 9, 0]:
            cache.access(position)

        self.assertEqual(cache.stats()["sequential_predictions"], 1)
        self.assertIn(1, cache.prefetched)

    def test_sequential_prefetches_can_use_full_cache(self):
        cache = PredictiveCache(
            capacity=3,
            disk_size=20,
            sequential_min_length=3,
        )

        outcomes = [cache.access(position) for position in range(10)]
        stats = cache.stats()

        self.assertEqual(outcomes[:3], [False, False, False])
        self.assertEqual(outcomes[3:], [True] * 7)
        self.assertEqual(stats["useful_prefetches"], 7)
        self.assertEqual(stats["demand_evictions_for_prefetches"], 8)
        self.assertEqual(stats["skipped_prefetches"], 0)


if __name__ == "__main__":
    unittest.main()
