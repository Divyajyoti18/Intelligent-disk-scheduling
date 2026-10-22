def main():

    from simulator.cache import DiskCache
    from simulator.predictive_cache import PredictiveCache


    def run_test(name, requests, capacity=3, disk_size=200):
        print(f"\n{'=' * 15} {name} {'=' * 15}")

        caches = {
            "LRU": DiskCache(capacity=capacity),
            "Predictive": PredictiveCache(
                capacity=capacity,
                history_size=3,
                min_confidence=0.6,
                min_observations=2,
                disk_size=disk_size,
            ),
        }

        for cache_name, cache in caches.items():
            for position in requests:
                cache.access(position)

            print(f"\n{cache_name}")
            for metric, value in cache.stats().items():
                print(f"{metric}: {value}")


    # Test 1: Consecutive disk positions
    sequential_requests = list(range(50, 90))

    # Test 2: Repeating sequence with six unique positions
    repeating_requests = [10, 20, 30, 40, 50, 60] * 20

    run_test("Sequential access", sequential_requests)
    run_test("Repeating pattern", repeating_requests)


if __name__ == '__main__':
    main()
