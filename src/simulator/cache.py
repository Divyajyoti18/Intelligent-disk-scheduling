
from collections import OrderedDict


class DiskCache:
    def __init__(self, capacity=5):
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive")

        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def access(self, position):
        if position in self.cache:
            self.hits += 1
            self.cache.move_to_end(position)
            return True

        self.misses += 1

        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            self.evictions += 1

        self.cache[position] = True
        return False

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate
        }
