
from schedulers.algorithms import (
    fcfs,
    sstf,
    scan,
    cscan,
    look,
    clook,
)


class DiskSchedulingEnvironment:
    """Contextual environment for selecting a disk scheduling algorithm."""

    def __init__(self, disk_size=200):
        if disk_size <= 0:
            raise ValueError("Disk size must be positive")

        self.disk_size = disk_size

        self.algorithms = {
            "FCFS": lambda requests, head: fcfs(requests, head),
            "SSTF": lambda requests, head: sstf(requests, head),
            "SCAN": lambda requests, head: scan(
                requests, head, self.disk_size
            ),
            "C-SCAN": lambda requests, head: cscan(
                requests, head, self.disk_size
            ),
            "LOOK": lambda requests, head: look(requests, head),
            "C-LOOK": lambda requests, head: clook(requests, head),
        }

    
    def get_state(self, requests, initial_head):
        """Classify a workload using trend, locality, and request distribution."""

        if not 0 <= initial_head < self.disk_size:
            raise ValueError("Initial head outside disk boundaries")

        if not requests:
            return "empty"

        if any(
            position < 0 or position >= self.disk_size
            for position in requests
        ):
            raise ValueError("Request outside disk boundaries")

        # 1. Detect strongly increasing or decreasing request sequences.
        if len(requests) >= 3:
            transitions = len(requests) - 1

            increasing = sum(
                requests[i] > requests[i - 1]
                for i in range(1, len(requests))
            )

            decreasing = sum(
                requests[i] < requests[i - 1]
                for i in range(1, len(requests))
            )

            if increasing / transitions >= 0.8:
                return "increasing"

            if decreasing / transitions >= 0.8:
                return "decreasing"

        # 2. Calculate workload distribution.
        minimum = min(requests)
        maximum = max(requests)
        spread = maximum - minimum
        center = (minimum + maximum) / 2

        # 3. Detect localized requests and their position relative
        #    to the current disk head.
        if spread <= self.disk_size * 0.25:
            head_distance = abs(center - initial_head)

            if head_distance <= self.disk_size * 0.15:
                return "localized_near_head"

            if center < initial_head:
                return "localized_left"

            return "localized_right"

        # 4. Detect bursty workloads: requests span a broad region,
        #    but consecutive requests tend to have relatively
        #    small jumps.
        distances = [
            abs(requests[i] - requests[i - 1])
            for i in range(1, len(requests))
        ]

        if distances:
            average_jump = sum(distances) / len(distances)

            if (
                spread > self.disk_size * 0.4
                and average_jump < self.disk_size * 0.15
            ):
                return "bursty"

        # 5. Distinguish workloads dominated by requests on one side
        #    of the current head from balanced random workloads.
        right_count = sum(
            position >= initial_head for position in requests
        )
        right_ratio = right_count / len(requests)

        if right_ratio >= 0.7:
            return "random_right"

        if right_ratio <= 0.3:
            return "random_left"

        return "random_balanced"


    def step(self, requests, initial_head, action):
        """Evaluate an action and return its relative reward and metrics."""

        if action not in self.algorithms:
            raise ValueError(f"Unknown scheduling action: {action}")

        requests = list(requests)
        state = self.get_state(requests, initial_head)

        # Evaluate every algorithm on the identical request trace.
        results = {}

        for name, algorithm in self.algorithms.items():
            order, movement = algorithm(
                list(requests), initial_head
            )
            results[name] = {
                "order": order,
                "movement": movement,
            }

        chosen = results[action]
        movements = [
            result["movement"]
            for result in results.values()
        ]

        best_movement = min(movements)
        worst_movement = max(movements)
        chosen_movement = chosen["movement"]

        # Reward 1 for the best movement and 0 for the worst.
        # If all algorithms tie, every action receives reward 1.
        if worst_movement == best_movement:
            reward = 1.0
        else:
            reward = 1.0 - (
                (chosen_movement - best_movement)
                / (worst_movement - best_movement)
            )

        metrics = {
            "algorithm": action,
            "head_movement": chosen_movement,
            "requests_serviced": len(chosen["order"]),
            "best_movement": best_movement,
            "worst_movement": worst_movement,
            "best_algorithms": [
                name for name, result in results.items()
                if result["movement"] == best_movement
            ],
            "reward": reward,
        }

        return state, reward, metrics
