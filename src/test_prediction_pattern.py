
from simulator.cache import DiskCache
from simulator.predictive_cache import PredictiveCache

# Repeated pattern with three positions.
requests = [10, 20, 30, 10, 20, 30, 10, 20, 30] * 5

caches = {
    "LRU": DiskCache(capacity=2),
    "Predictive": PredictiveCache(capacity=2, min_confidence=0.5)
}

for name, cache in caches.items():
    for position in requests:
        cache.access(position)

    print(f"\n{name}")
    print(cache.stats())

