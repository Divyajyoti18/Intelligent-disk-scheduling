
import random
from collections import defaultdict


class QLearningAgent:
    """Tabular Q-learning agent for selecting disk scheduling algorithms."""

    def __init__(
        self,
        actions=("FCFS", "SSTF", "SCAN", "LOOK"),
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        seed=42,
    ):
        if not actions:
            raise ValueError("At least one action is required")

        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be in (0, 1]")

        if not 0 <= discount_factor <= 1:
            raise ValueError("Discount factor must be in [0, 1]")

        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be in [0, 1]")

        self.actions = tuple(actions)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table = defaultdict(
            lambda: {action: 0.0 for action in self.actions}
        )
        self.rng = random.Random(seed)

    def choose_action(self, state, explore=True):
        """Choose an action using epsilon-greedy exploration."""
        if explore and self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)

        values = self.q_table[state]
        best_value = max(values.values())

        best_actions = [
            action
            for action, value in values.items()
            if value == best_value
        ]

        return self.rng.choice(best_actions)

    def update(self, state, action, reward, next_state, done=False):
        """Update the Q-value from one transition."""
        current_q = self.q_table[state][action]

        if done:
            future_q = 0.0
        else:
            future_q = max(self.q_table[next_state].values())

        target = reward + self.discount_factor * future_q

        self.q_table[state][action] = current_q + (
            self.learning_rate * (target - current_q)
        )

    def decay_exploration(self):
        """Gradually shift from exploration toward exploitation."""
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def best_action(self, state):
        """Return the best currently learned action without exploration."""
        return self.choose_action(state, explore=False)
