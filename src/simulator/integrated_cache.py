
from simulator.predictive_cache import PredictiveCache


class IntegratedPredictiveCache(PredictiveCache):
    """
    Predictive cache for the integrated disk simulator.

    A cache miss is recorded immediately, but the requested
    position is inserted only after the disk services it.
    """

    def demand_lookup(self, position):
        """Check the cache without inserting a missed request."""
        if position in self.cache:
            self.hits += 1

            if position in self.prefetched:
                self.prefetched.remove(position)
                self.useful_prefetches += 1

            self.cache.move_to_end(position)
            return True

        self.misses += 1
        return False

    def observe_and_predict(self, position):
        """
        Update the request-pattern model and return the next
        predicted position, if confidence requirements are met.
        """

        # Evaluate the previous prediction.
        if self.pending_prediction is not None:
            correct = self.pending_prediction == position

            self.prediction_attempts += 1
            self.correct_predictions += int(correct)
            self.recent_outcomes.append(int(correct))

            if len(self.recent_outcomes) > self.adaptive_window:
                self.recent_outcomes.pop(0)

            self.pending_prediction = None

        # Learn transitions from the previous request history.
        for order in range(
            1, min(self.history_size, len(self.history)) + 1
        ):
            context = tuple(self.history[-order:])
            self.transitions[context][position] += 1

        self.history.append(position)

        history_limit = max(
            self.history_size, self.sequential_min_length
        )

        if len(self.history) > history_limit:
            self.history.pop(0)

        # Prefer a detected sequential pattern.
        predicted = self._predict_sequential()

        if predicted is not None:
            prediction_type = "sequential"
        else:
            predicted = self._predict_next()
            prediction_type = "transition"

        if predicted is None:
            return None

        self.pending_prediction = predicted

        if prediction_type == "sequential":
            self.sequential_predictions += 1
        else:
            self.transition_predictions += 1

        return predicted

    def complete_disk_read(self, position, is_prefetch=False):
        """Insert a block after its disk read has completed."""
        self._insert(position, is_prefetch=is_prefetch)
