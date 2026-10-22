
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from rl.q_learning import QLearningAgent
from rl.environment import DiskSchedulingEnvironment
from simulator.cache import DiskCache
from simulator.predictive_cache import PredictiveCache
from simulator.workload import (
    random_workload,
    sequential_workload,
    localized_workload,
    bursty_workload,
)


# ==================== CONFIGURATION ====================

DISK_SIZE = 200
REQUEST_COUNT = 100
TRIALS = 30
CACHE_CAPACITY = 10
INITIAL_HEAD = 100
SEED = 42

ACTIONS = (
    "FCFS", "SSTF", "SCAN",
    "C-SCAN", "LOOK", "C-LOOK",
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

environment = DiskSchedulingEnvironment(DISK_SIZE)


# ==================== LOAD TRAINED RL MODEL ====================

model_path = ROOT / "src" / "rl" / "q_table.json"

if not model_path.exists():
    raise FileNotFoundError(
        f"Trained Q-table not found: {model_path}\n"
        "Run python src/train_rl_scheduler.py first."
    )

agent = QLearningAgent(
    actions=ACTIONS,
    seed=SEED,
)

with model_path.open("r", encoding="utf-8") as file:
    saved_q_table = json.load(file)

# Restore learned action values without retraining.
for state, values in saved_q_table.items():
    agent.q_table[state] = {
        action: float(values.get(action, 0.0))
        for action in ACTIONS
    }


# ==================== WORKLOAD GENERATION ====================

def generate_workload(kind, rng):
    if kind == "Random":
        return random_workload(REQUEST_COUNT, DISK_SIZE)

    if kind == "Sequential":
        return sequential_workload(
            REQUEST_COUNT, start=50
        )

    if kind == "Localized":
        return localized_workload(
            REQUEST_COUNT,
            center=rng.randint(40, 160),
            spread=rng.randint(8, 30),
            disk_size=DISK_SIZE,
        )

    if kind == "Bursty":
        return bursty_workload(
            REQUEST_COUNT, DISK_SIZE
        )

    raise ValueError(f"Unknown workload: {kind}")


WORKLOADS = [
    "Random",
    "Sequential",
    "Localized",
    "Bursty",
]


# ==================== INTEGRATED EVALUATION ====================

trial_results = []
grouped_results = defaultdict(list)

print("INTEGRATED RL SCHEDULING + PREDICTIVE CACHING")
print(f"Trials per workload: {TRIALS}")
print(f"Requests per trial: {REQUEST_COUNT}")
print(f"Cache capacity: {CACHE_CAPACITY}")
print(f"Disk size: {DISK_SIZE}")
print()

for workload_index, workload in enumerate(WORKLOADS):
    for trial in range(1, TRIALS + 1):

        # A reproducible, shared workload trace.
        trial_seed = (
            SEED + workload_index * 10000 + trial
        )
        rng = random.Random(trial_seed)
        random.seed(trial_seed)

        requests = generate_workload(workload, rng)

        # RL selects an algorithm from the learned Q-table.
        state = environment.get_state(
            requests, INITIAL_HEAD
        )
        selected_algorithm = agent.best_action(state)

        # Evaluate the selected algorithm and FCFS baseline.
        _, fcfs_movement = environment.algorithms["FCFS"](
            list(requests), INITIAL_HEAD
        )

        _, rl_movement = environment.algorithms[
            selected_algorithm
        ](list(requests), INITIAL_HEAD)

        scheduler_results = {
            "FCFS": {
                "algorithm": "FCFS",
                "head_movement": fcfs_movement,
            },
            "RL": {
                "algorithm": selected_algorithm,
                "head_movement": rl_movement,
            },
        }

        # Four combinations: two schedulers x two caches.
        for scheduler_name in ("FCFS", "RL"):
            scheduler = scheduler_results[scheduler_name]

            for cache_name in ("LRU", "Predictive"):

                # A fresh cache for every configuration and trial.
                if cache_name == "LRU":
                    cache = DiskCache(
                        capacity=CACHE_CAPACITY
                    )
                else:
                    cache = PredictiveCache(
                        capacity=CACHE_CAPACITY,
                        history_size=3,
                        min_confidence=0.6,
                        min_observations=2,
                        disk_size=DISK_SIZE,
                        sequential_min_length=3,
                    )

                # Cache observes incoming requests in arrival order.
                for position in requests:
                    cache.access(position)

                stats = cache.stats()

                row = {
                    "workload": workload,
                    "trial": trial,
                    "state": state,
                    "scheduler": scheduler_name,
                    "selected_algorithm": scheduler["algorithm"],
                    "cache": cache_name,
                    "head_movement": scheduler["head_movement"],
                    "cache_hits": stats["hits"],
                    "cache_misses": stats["misses"],
                    "cache_hit_rate": stats["hit_rate"],
                    "prefetches": stats.get("prefetches", 0),
                    "useful_prefetches": stats.get(
                        "useful_prefetches", 0
                    ),
                    "wasted_prefetches": stats.get(
                        "wasted_prefetches", 0
                    ),
                }

                trial_results.append(row)

                key = (workload, scheduler_name, cache_name)
                grouped_results[key].append(row)


# ==================== SUMMARY ====================

summary_results = []

for (workload, scheduler_name, cache_name), rows in (
    grouped_results.items()
):
    summary = {
        "workload": workload,
        "scheduler": scheduler_name,
        "cache": cache_name,
        "trials": len(rows),
    }

    metrics = [
        "head_movement",
        "cache_hits",
        "cache_misses",
        "cache_hit_rate",
        "prefetches",
        "useful_prefetches",
        "wasted_prefetches",
    ]

    for metric in metrics:
        summary[f"avg_{metric}"] = statistics.mean(
            row[metric] for row in rows
        )

    summary_results.append(summary)


# ==================== PRINT RESULTS ====================

print(
    f"{'Workload':12} | {'Scheduler':9} | "
    f"{'Cache':10} | {'Movement':9} | "
    f"{'Hit Rate':9} | {'Useful PF':9}"
)
print("-" * 85)

for row in summary_results:
    print(
        f"{row['workload']:12} | "
        f"{row['scheduler']:9} | "
        f"{row['cache']:10} | "
        f"{row['avg_head_movement']:9.2f} | "
        f"{row['avg_cache_hit_rate']:8.2%} | "
        f"{row['avg_useful_prefetches']:9.2f}"
    )


# ==================== SAVE CSV ====================

def save_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


trials_path = RESULTS_DIR / "integrated_trials.csv"
summary_path = RESULTS_DIR / "integrated_summary.csv"

save_csv(trials_path, trial_results)
save_csv(summary_path, summary_results)


# ==================== GRAPH 1: HEAD MOVEMENT ====================

fig, ax = plt.subplots(figsize=(10, 6))

x = list(range(len(WORKLOADS)))
width = 0.35

for index, scheduler_name in enumerate(("FCFS", "RL")):
    values = []

    for workload in WORKLOADS:
        rows = [
            row for row in summary_results
            if row["workload"] == workload
            and row["scheduler"] == scheduler_name
            and row["cache"] == "Predictive"
        ]
        values.append(rows[0]["avg_head_movement"])

    offsets = [
        position + (index - 0.5) * width
        for position in x
    ]

    ax.bar(
        offsets,
        values,
        width,
        label=scheduler_name,
    )

ax.set_xticks(x)
ax.set_xticklabels(WORKLOADS)
ax.set_xlabel("Workload")
ax.set_ylabel("Average Head Movement")
ax.set_title("Integrated System: FCFS vs RL Scheduling")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

movement_path = FIGURES_DIR / "integrated_head_movement.png"
fig.savefig(movement_path, dpi=200)
plt.close(fig)


# ==================== GRAPH 2: CACHE HIT RATE ====================

fig, ax = plt.subplots(figsize=(10, 6))

for index, cache_name in enumerate(("LRU", "Predictive")):
    values = []

    for workload in WORKLOADS:
        rows = [
            row for row in summary_results
            if row["workload"] == workload
            and row["scheduler"] == "RL"
            and row["cache"] == cache_name
        ]
        values.append(
            rows[0]["avg_cache_hit_rate"] * 100
        )

    offsets = [
        position + (index - 0.5) * width
        for position in x
    ]

    ax.bar(
        offsets,
        values,
        width,
        label=cache_name,
    )

ax.set_xticks(x)
ax.set_xticklabels(WORKLOADS)
ax.set_xlabel("Workload")
ax.set_ylabel("Average Cache Hit Rate (%)")
ax.set_title("RL Scheduler: LRU vs Predictive Cache")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

cache_path = FIGURES_DIR / "integrated_cache_hit_rate.png"
fig.savefig(cache_path, dpi=200)
plt.close(fig)


# ==================== FINAL REPORT ====================

print("\nIntegration evaluation completed.")

print(f"Detailed CSV: {trials_path}")
print(f"Summary CSV:  {summary_path}")

print(f"Head movement graph: {movement_path}")
print(f"Cache hit-rate graph: {cache_path}")
