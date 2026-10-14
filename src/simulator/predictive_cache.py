
from collections import Counter, OrderedDict


class PredictiveCache:
    def __init__(self, capacity=5, history_size=3):
        if capacity <= 0 or history_size <= 0:
            raise ValueError("Capacity and history size must be positive")

        self.capacity = capacity
        self.history_size = history_size
        self.cache = OrderedDict()
        self.prefetched = set()
        self.history = []
        self.transitions = {}
        self.hits = 0
        self.misses = 0
        self.prefetches = 0
        self.useful_prefetches = 0
        self.evictions = 0

    def _insert(self, position, is_prefetch=False):
        if position in self.cache:
            self.cache.move_to_end(position)
            return

        if len(self.cache) >= self.capacity:
            removed, _ = self.cache.popitem(last=False)
            self.prefetched.discard(removed)
            self.evictions += 1

        self.cache[position] = True

        if is_prefetch:
            self.prefetched.add(position)

    def _predict_next(self, current):
        next_positions = self.transitions.get(current)

        if not next_positions:
            return None

        return next_positions.most_common(1)[0][0]

    def access(self, position):
        hit = position in self.cache

        if position in self.cache:
            self.hits += 1

            if position in self.prefetched:
                self.useful_prefetches += 1
                self.prefetched.remove(position)

            self.cache.move_to_end(position)
        else:
            self.misses += 1
            self._insert(position)

        if self.history:
            previous = self.history[-1]
            counts = self.transitions.setdefault(previous, Counter())
            counts[position] += 1

        self.history.append(position)

        if len(self.history) > self.history_size:
            self.history.pop(0)

        predicted = self._predict_next(position)

        if predicted is not None and predicted not in self.cache:
            self._insert(predicted, is_prefetch=True)
            self.prefetches += 1

        return hit

    def stats(self):
        total = self.hits + self.misses

        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "prefetches": self.prefetches,
            "useful_prefetches": self.useful_prefetches,
            "hit_rate": self.hits / total if total else 0
        }
