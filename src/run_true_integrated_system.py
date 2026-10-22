
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Ensure imports work when this script is run directly.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rl.environment import DiskSchedulingEnvironment
from rl.q_learning import QLearningAgent
from simulator.cache import DiskCache
from simulator.integrated_cache import IntegratedPredictiveCache
from simulator.workload import (
    random_workload,
    sequential_workload,
    localized_workload,
    bursty_workload,
)
from schedulers.algorithms import (
    fcfs,
    sstf,
    scan,
    cscan,
    look,
    clook,
)


DISK_SIZE = 200
INITIAL_HEAD = 53
CACHE_CAPACITY = 5
REQUESTS_PER_TRIAL = 100
BATCH_SIZE = 10
TRIALS = 30

RESULTS_DIR = ROOT.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

WORKLOADS = {
    "Random": random_workload,
    "Sequential": sequential_workload,
    "Localized": localized_workload,
    "Bursty": bursty_workload,
}


def load_agent():
    """Load the trained Q-table and configure greedy evaluation."""
    q_table_path = ROOT / "rl" / "q_table.json"

    if not q_table_path.exists():
        raise FileNotFoundError(
            f"Trained Q-table not found: {q_table_path}\n"
            "Run src/train_rl_scheduler.py first."
        )

    with open(q_table_path, "r", encoding="utf-8") as file:
        saved_table = json.load(file)

    # Use the action names actually present in the saved Q-table.
    actions = tuple(
        next(iter(saved_table.values())).keys()
    ) if saved_table else ("FCFS", "SSTF", "SCAN", "LOOK")

    agent = QLearningAgent(actions=actions, epsilon=0.0)

    for state, values in saved_table.items():
        agent.q_table[state] = values

    return agent


def schedule_requests(requests, head, action):
    """Run one scheduling algorithm on the pending disk requests."""
    action = action.upper()

    if not requests:
        return [], 0

    if action == "FCFS":
        return fcfs(requests, head)

    if action == "SSTF":
        return sstf(requests, head)

    if action == "SCAN":
        return scan(requests, head, DISK_SIZE, "right")

    if action == "C-SCAN":
        return cscan(requests, head, DISK_SIZE)

    if action == "LOOK":
        return look(requests, head, "right")

    if action == "C-LOOK":
        return clook(requests, head)

    raise ValueError(f"Unsupported scheduling action: {action}")


def physical_end_position(requests, head, action):
    """Return the actual head position after an algorithm finishes."""
    if not requests:
        return head

    action = action.upper()
    left = [p for p in requests if p < head]
    right = [p for p in requests if p >= head]

    if action == "SCAN":
        if left:
            return min(left)
        return DISK_SIZE - 1

    if action == "C-SCAN":
        if left:
            return max(left)
        return 0

    if action == "C-LOOK":
        if left:
            return max(left)
        return max(right)

    # FCFS, SSTF, LOOK
    order, _ = schedule_requests(requests, head, action)
    return order[-1] if order else head




def run_trial(requests, agent, cache_name, scheduler_name):
    """Run one trial with explicit cache and physical-head accounting."""

    if cache_name == "LRU":
        cache = DiskCache(capacity=CACHE_CAPACITY)
    else:
        cache = IntegratedPredictiveCache(
            capacity=CACHE_CAPACITY,
            disk_size=DISK_SIZE,
        )

    head = INITIAL_HEAD
    total_movement = 0
    selected_actions = []

    for start in range(0, len(requests), BATCH_SIZE):
        batch = requests[start:start + BATCH_SIZE]
        pending = []
        pending_set = set()
        predicted_positions = []

        # 1. Process incoming demand requests in arrival order.
        for position in batch:

            if cache_name == "LRU":
                hit = cache.access(position)

                # DiskCache inserts misses immediately. A repeated
                # request in this batch will therefore become a hit.
                if not hit and position not in pending_set:
                    pending.append(position)
                    pending_set.add(position)

            else:
                # Always record the demand lookup, even if the
                # same block is already waiting for disk service.
                hit = cache.demand_lookup(position)

                if not hit and position not in pending_set:
                    pending.append(position)
                    pending_set.add(position)

                predicted = cache.observe_and_predict(position)

                if predicted is not None:
                    # Only prefetch positions that are not requested
                    # in the current batch. They may benefit future batches.
                    if (
                        predicted not in cache.cache
                        and predicted not in pending_set
                        and predicted not in batch
                        and predicted not in predicted_positions
                    ):
                        predicted_positions.append(predicted)
                        
        # 2. Select a scheduling algorithm for demand misses.
        if scheduler_name == "FCFS":
            action = "FCFS"
        else:
            state = DiskSchedulingEnvironment(
                DISK_SIZE
            ).get_state(pending, head)

            action = agent.best_action(state)

        selected_actions.append(action)

        # 3. Service demand misses.
        order, movement = schedule_requests(pending, head, action)
        total_movement += movement

        head = physical_end_position(pending, head, action)

        # 4. Insert demand blocks after their reads complete.
        if cache_name == "Predictive":
            for position in order:
                cache.complete_disk_read(position)

        # 5. Perform actual prefetch reads.
        # These operations also move the physical disk head.
        if cache_name == "Predictive":
            for position in predicted_positions:
                if position in cache.cache:
                    continue

                _, movement = schedule_requests(
                    [position], head, "FCFS"
                )

                total_movement += movement
                head = position

                cache.complete_disk_read(
                    position,
                    is_prefetch=True,
                )
                cache.prefetches += 1

    stats = cache.stats()

    return {
        "head_movement": total_movement,
        "cache_hit_rate": stats["hit_rate"],
        "cache_hits": stats["hits"],
        "cache_misses": stats["misses"],
        "prefetches": stats.get("prefetches", 0),
        "useful_prefetches": stats.get("useful_prefetches", 0),
        "wasted_prefetches": stats.get("wasted_prefetches", 0),
        "prediction_accuracy": stats.get("prediction_accuracy", 0),
        "selected_actions": ",".join(selected_actions),
    }



def generate_workload(generator):
    """Generate one trace using the existing workload function."""
    return generator(REQUESTS_PER_TRIAL, DISK_SIZE)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    agent = load_agent()
    rows = []

    for workload_name, generator in WORKLOADS.items():
        for trial in range(TRIALS):
            requests = generate_workload(generator)

            for scheduler_name in ("FCFS", "RL"):
                for cache_name in ("LRU", "Predictive"):
                    result = run_trial(
                        requests,
                        agent,
                        cache_name,
                        scheduler_name,
                    )

                    rows.append({
                        "workload": workload_name,
                        "trial": trial + 1,
                        "scheduler": scheduler_name,
                        "cache": cache_name,
                        **result,
                    })

    csv_path = RESULTS_DIR / "true_integrated_trials.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Summarize the average performance for each configuration.
    summary = {}

    for row in rows:
        key = (row["workload"], row["scheduler"], row["cache"])
        summary.setdefault(key, []).append(row)

    summary_rows = []

    for (workload, scheduler, cache_name), group in summary.items():
        summary_rows.append({
            "workload": workload,
            "scheduler": scheduler,
            "cache": cache_name,
            "avg_head_movement": sum(
                r["head_movement"] for r in group
            ) / len(group),
            "avg_cache_hit_rate": sum(
                r["cache_hit_rate"] for r in group
            ) / len(group),
            "avg_prediction_accuracy": sum(
                r["prediction_accuracy"] for r in group
            ) / len(group),
            "avg_useful_prefetches": sum(
                r["useful_prefetches"] for r in group
            ) / len(group),
        })

    summary_path = RESULTS_DIR / "true_integrated_summary.csv"

    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=summary_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Plot average head movement by workload and configuration.
    for workload in WORKLOADS:
        entries = [
            row for row in summary_rows
            if row["workload"] == workload
        ]

        labels = [
            f'{row["scheduler"]}\n{row["cache"]}'
            for row in entries
        ]
        values = [row["avg_head_movement"] for row in entries]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values)
        plt.title(f"Integrated System: {workload} Workload")
        plt.ylabel("Average Head Movement")
        plt.tight_layout()

        filename = workload.lower() + "_integrated.png"
        plt.savefig(FIGURES_DIR / filename)
        plt.close()

    print("Integrated simulation completed.")
    print(f"Trial results:   {csv_path}")
    print(f"Summary results: {summary_path}")
    print(f"Figures saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
