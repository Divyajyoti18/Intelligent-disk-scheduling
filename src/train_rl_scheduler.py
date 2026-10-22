
import csv
import json
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from rl.q_learning import QLearningAgent
from rl.environment import DiskSchedulingEnvironment
from simulator.workload import (
    random_workload,
    sequential_workload,
    localized_workload,
    bursty_workload,
)


# ==================== CONFIGURATION ====================

DISK_SIZE = 200
REQUEST_COUNT = 100
EPISODES = 12000
TRIALS = 30
INITIAL_HEAD = 100
SEED = 42

ACTIONS = (
    "FCFS", "SSTF", "SCAN",
    "C-SCAN", "LOOK", "C-LOOK",
)

WORKLOAD_TYPES = [
    "Random", "Sequential", "Localized", "Bursty"
]

# Project paths.
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
rng = random.Random(SEED)

environment = DiskSchedulingEnvironment(DISK_SIZE)

agent = QLearningAgent(
    actions=ACTIONS,
    learning_rate=0.1,
    discount_factor=0.0,
    epsilon=1.0,
    epsilon_decay=0.9995,
    min_epsilon=0.05,
    seed=SEED,
)


# ==================== WORKLOAD GENERATION ====================

def generate_workload(kind):
    if kind == "Random":
        return random_workload(REQUEST_COUNT, DISK_SIZE)

    if kind == "Sequential":
        return sequential_workload(REQUEST_COUNT, start=50)

    if kind == "Localized":
        return localized_workload(
            REQUEST_COUNT,
            center=rng.randint(40, 160),
            spread=rng.randint(8, 30),
            disk_size=DISK_SIZE,
        )

    if kind == "Bursty":
        return bursty_workload(REQUEST_COUNT, DISK_SIZE)

    raise ValueError(f"Unknown workload: {kind}")


# ==================== TRAINING ====================

for episode in range(EPISODES):
    kind = rng.choice(WORKLOAD_TYPES)
    requests = generate_workload(kind)

    state = environment.get_state(requests, INITIAL_HEAD)
    action = agent.choose_action(state)

    _, reward, _ = environment.step(
        requests, INITIAL_HEAD, action
    )

    agent.update(
        state=state,
        action=action,
        reward=reward,
        next_state=state,
        done=True,
    )

    agent.decay_exploration()


# Save the trained Q-table.
model_path = ROOT / "src" / "rl" / "q_table.json"
model_path.parent.mkdir(parents=True, exist_ok=True)

with model_path.open("w", encoding="utf-8") as file:
    json.dump(dict(agent.q_table), file, indent=2)

print("\nTraining completed.")
print(f"Episodes: {EPISODES}")
print(f"Learned states: {len(agent.q_table)}")
print(f"Final exploration rate: {agent.epsilon:.3f}")
print(f"Q-table saved to: {model_path}")


# ==================== EVALUATION ====================

print(f"\nEvaluation: {TRIALS} trials per workload")

trial_results = []
summary_results = []

for kind in WORKLOAD_TYPES:
    rl_movements = []
    oracle_movements = []
    regrets = []
    optimal_count = 0

    algorithm_movements = {
        action: [] for action in ACTIONS
    }

    for trial in range(1, TRIALS + 1):
        # Every algorithm receives the exact same request sequence.
        requests = generate_workload(kind)

        state = environment.get_state(requests, INITIAL_HEAD)
        selected = agent.best_action(state)

        # Measure all traditional algorithms on this trace.
        all_movements = {}

        for action in ACTIONS:
            _, _, metrics = environment.step(
                requests, INITIAL_HEAD, action
            )

            movement = metrics["head_movement"]
            all_movements[action] = movement
            algorithm_movements[action].append(movement)

        rl_movement = all_movements[selected]
        oracle_movement = min(all_movements.values())
        regret = rl_movement - oracle_movement

        is_optimal = rl_movement == oracle_movement

        rl_movements.append(rl_movement)
        oracle_movements.append(oracle_movement)
        regrets.append(regret)

        if is_optimal:
            optimal_count += 1

        # Store the complete result of each trial.
        row = {
            "workload": kind,
            "trial": trial,
            "state": state,
            "selected_algorithm": selected,
            "rl_movement": rl_movement,
            "oracle_movement": oracle_movement,
            "regret": regret,
            "is_optimal": is_optimal,
        }

        for action in ACTIONS:
            row[f"movement_{action}"] = all_movements[action]

        trial_results.append(row)

    # Calculate averages for this workload.
    summary = {
        "workload": kind,
        "trials": TRIALS,
        "avg_rl_movement": statistics.mean(rl_movements),
        "avg_oracle_movement": statistics.mean(oracle_movements),
        "avg_regret": statistics.mean(regrets),
        "optimal_selection_pct": 100 * optimal_count / TRIALS,
    }

    for action in ACTIONS:
        summary[f"avg_{action}_movement"] = statistics.mean(
            algorithm_movements[action]
        )

    summary_results.append(summary)


# ==================== SAVE CSV FILES ====================

trials_path = RESULTS_DIR / "rl_evaluation_trials.csv"

with trials_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=trial_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(trial_results)


summary_path = RESULTS_DIR / "rl_evaluation_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=summary_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary_results)


# ==================== PRINT SUMMARY ====================

print("\nEvaluation summary")
print(
    f"{'Workload':12} | {'RL Movement':12} | "
    f"{'Oracle':10} | {'Regret':10} | {'Optimal %':10}"
)
print("-" * 72)

for row in summary_results:
    print(
        f"{row['workload']:12} | "
        f"{row['avg_rl_movement']:12.2f} | "
        f"{row['avg_oracle_movement']:10.2f} | "
        f"{row['avg_regret']:10.2f} | "
        f"{row['optimal_selection_pct']:9.2f}%"
    )


# ==================== GRAPH 1: HEAD MOVEMENT ====================

workload_names = WORKLOAD_TYPES
series_names = list(ACTIONS) + ["RL policy"]

x_positions = list(range(len(workload_names)))
bar_width = 0.8 / len(series_names)

fig, ax = plt.subplots(figsize=(15, 7))

for index, series_name in enumerate(series_names):
    if series_name == "RL policy":
        values = [
            row["avg_rl_movement"] for row in summary_results
        ]
    else:
        values = [
            row[f"avg_{series_name}_movement"]
            for row in summary_results
        ]

    offsets = [
        x + (index - (len(series_names) - 1) / 2) * bar_width
        for x in x_positions
    ]

    ax.bar(
        offsets,
        values,
        bar_width,
        label=series_name,
    )

ax.set_xticks(x_positions)
ax.set_xticklabels(workload_names)
ax.set_xlabel("Workload")
ax.set_ylabel("Average Head Movement (cylinders)")
ax.set_title("Disk Scheduling: Traditional Algorithms vs RL")
ax.legend(ncol=4)
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

head_chart = FIGURES_DIR / "rl_head_movement.png"
fig.savefig(head_chart, dpi=200)
plt.close(fig)


# ==================== GRAPH 2: AVERAGE REGRET ====================

fig, ax = plt.subplots(figsize=(9, 6))

regret_values = [
    row["avg_regret"] for row in summary_results
]

bars = ax.bar(workload_names, regret_values)

ax.bar_label(bars, fmt="%.2f", padding=3)
ax.set_xlabel("Workload")
ax.set_ylabel("Average Regret (cylinders)")
ax.set_title("RL Scheduling Regret by Workload")
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.set_ylim(
    0,
    max(regret_values) * 1.2 if max(regret_values) > 0 else 1,
)

fig.tight_layout()

regret_chart = FIGURES_DIR / "rl_regret.png"
fig.savefig(regret_chart, dpi=200)
plt.close(fig)


# ==================== GRAPH 3: OPTIMAL SELECTION ====================

fig, ax = plt.subplots(figsize=(9, 6))

optimal_values = [
    row["optimal_selection_pct"] for row in summary_results
]

bars = ax.bar(workload_names, optimal_values)

ax.bar_label(bars, fmt="%.2f%%", padding=3)
ax.set_xlabel("Workload")
ax.set_ylabel("Optimal Selections (%)")
ax.set_title("RL Policy: Optimal Algorithm Selection Rate")
ax.set_ylim(0, 110)
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

optimal_chart = FIGURES_DIR / "rl_optimal_selection.png"
fig.savefig(optimal_chart, dpi=200)
plt.close(fig)


# ==================== COMPLETION REPORT ====================

print("\nCSV files saved:")
print(f"Detailed trials: {trials_path}")
print(f"Summary:         {summary_path}")

print("\nGraphs saved:")
print(f"Head movement:   {head_chart}")
print(f"Average regret:  {regret_chart}")
print(f"Optimal rate:    {optimal_chart}")
