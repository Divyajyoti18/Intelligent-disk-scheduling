
from collections import Counter, OrderedDict, defaultdict


class PredictiveCache:
    def __init__(
        self,
        capacity=5,
        history_size=3,
        min_confidence=0.6,
        min_observations=2,
        disk_size=200,
        sequential_min_length=3,
        adaptive_window=10,
        adaptive_threshold=0.5,
    ):
        if capacity <= 0 or history_size <= 0:
            raise ValueError("Capacity and history size must be positive")
        if not 0 < min_confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if min_observations <= 0 or disk_size <= 0:
            raise ValueError("Observations and disk size must be positive")
        if sequential_min_length < 2:
            raise ValueError("Sequential minimum length must be at least 2")
        if adaptive_window <= 0:
            raise ValueError("Adaptive window must be positive")
        if not 0 <= adaptive_threshold <= 1:
            raise ValueError("Adaptive threshold must be between 0 and 1")

        self.capacity = capacity
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.min_observations = min_observations
        self.disk_size = disk_size
        self.sequential_min_length = sequential_min_length
        self.adaptive_window = adaptive_window
        self.adaptive_threshold = adaptive_threshold

        self.cache = OrderedDict()
        self.prefetched = set()
        self.history = []
        self.transitions = defaultdict(Counter)

        self.recent_outcomes = []
        self.pending_prediction = None

        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.prefetches = 0
        self.useful_prefetches = 0
        self.wasted_prefetches = 0
        self.demand_evictions_for_prefetches = 0

        self.prediction_attempts = 0
        self.correct_predictions = 0
        self.sequential_predictions = 0
        self.transition_predictions = 0
        self.skipped_prefetches = 0

    def _insert(self, position, is_prefetch=False):
        if position in self.cache:
            self.cache.move_to_end(position)
            return True

        if len(self.cache) >= self.capacity:
            # Prefer replacing an unused prefetch. If none exists, predictions
            # compete with demand data just like any other cache insertion.
            if is_prefetch:
                candidates = [
                    item for item in self.cache
                    if item in self.prefetched
                ]
                if candidates:
                    removed = candidates[0]
                    del self.cache[removed]
                    self.prefetched.remove(removed)
                    self.wasted_prefetches += 1
                else:
                    removed, _ = self.cache.popitem(last=False)
                    self.demand_evictions_for_prefetches += 1
            else:
                removed, _ = self.cache.popitem(last=False)
                if removed in self.prefetched:
                    self.prefetched.remove(removed)
                    self.wasted_prefetches += 1

            self.evictions += 1

        self.cache[position] = True
        if is_prefetch:
            self.prefetched.add(position)

        return True

    def _predict_sequential(self):
        length = self.sequential_min_length
        if len(self.history) < length:
            return None

        recent = self.history[-length:]

        for previous, current in zip(recent, recent[1:]):
            if (current - previous) % self.disk_size != 1:
                return None

        return (recent[-1] + 1) % self.disk_size

    def _predict_next(self):
        max_order = min(self.history_size, len(self.history))

        for order in range(max_order, 0, -1):
            context = tuple(self.history[-order:])
            candidates = self.transitions.get(context)

            if not candidates:
                continue

            next_position, count = candidates.most_common(1)[0]
            total = sum(candidates.values())
            confidence = count / total

            if (
                count >= self.min_observations
                and confidence >= self.min_confidence
            ):
                return next_position

        return None

    def _adaptive_prefetch_enabled(self):
        # Allow initial learning instead of requiring a full window.
        if len(self.recent_outcomes) < self.adaptive_window:
            return True

        accuracy = sum(self.recent_outcomes) / len(self.recent_outcomes)
        return accuracy >= self.adaptive_threshold

    def access(self, position):
        # Evaluate the previous next-request prediction.
        if self.pending_prediction is not None:
            correct = self.pending_prediction == position
            self.prediction_attempts += 1
            self.correct_predictions += int(correct)

            self.recent_outcomes.append(int(correct))
            if len(self.recent_outcomes) > self.adaptive_window:
                self.recent_outcomes.pop(0)

            self.pending_prediction = None

        hit = position in self.cache

        if hit:
            self.hits += 1
            if position in self.prefetched:
                self.prefetched.remove(position)
                self.useful_prefetches += 1
            self.cache.move_to_end(position)
        else:
            self.misses += 1
            self._insert(position)

        # Learn the actual next request for each available context.
        for order in range(1, min(self.history_size, len(self.history)) + 1):
            context = tuple(self.history[-order:])
            self.transitions[context][position] += 1

        self.history.append(position)
        history_limit = max(self.history_size, self.sequential_min_length)
        if len(self.history) > history_limit:
            self.history.pop(0)

        sequential_prediction = self._predict_sequential()
        if sequential_prediction is not None:
            predicted = sequential_prediction
            prediction_type = "sequential"
        else:
            predicted = self._predict_next()
            prediction_type = "transition"

        if predicted is not None:
            self.pending_prediction = predicted

            if prediction_type == "sequential":
                self.sequential_predictions += 1
            else:
                self.transition_predictions += 1

            if predicted not in self.cache:
                if self._adaptive_prefetch_enabled():
                    if self._insert(predicted, is_prefetch=True):
                        self.prefetches += 1
                else:
                    self.skipped_prefetches += 1

        return hit

    def stats(self):
        total = self.hits + self.misses
        attempts = self.prediction_attempts

        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "prefetches": self.prefetches,
            "useful_prefetches": self.useful_prefetches,
            "wasted_prefetches": self.wasted_prefetches,
            "demand_evictions_for_prefetches": (
                self.demand_evictions_for_prefetches
            ),
            "skipped_prefetches": self.skipped_prefetches,
            "hit_rate": self.hits / total if total else 0,
            "prediction_attempts": attempts,
            "correct_predictions": self.correct_predictions,
            "prediction_accuracy": (
                self.correct_predictions / attempts if attempts else 0
            ),
            "sequential_predictions": self.sequential_predictions,
            "transition_predictions": self.transition_predictions,
            "recent_prediction_accuracy": (
                sum(self.recent_outcomes) / len(self.recent_outcomes)
                if self.recent_outcomes else 0
            ),
        }
