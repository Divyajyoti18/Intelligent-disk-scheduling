
import csv
import json
import random
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

from schedulers.algorithms import fcfs, sstf, scan, cscan, look, clook
from simulator.workload import (
    random_workload,
    sequential_workload,
    localized_workload,
    bursty_workload,
)
from rl.environment import DiskSchedulingEnvironment


ROOT = Path(__file__).resolve().parent
DISK_SIZE = 200
REQUEST_COUNT = 100
TRIALS = 30
INITIAL_HEAD = 100

MODEL_PATH = ROOT / "rl" / "q_table.json"
RESULTS_DIR = ROOT.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

with MODEL_PATH.open(encoding="utf-8") as file:
    q_table = json.load(file)

environment = DiskSchedulingEnvironment(DISK_SIZE)

algorithms = {
    "FCFS": lambda r, h: fcfs(r, h),
    "SSTF": lambda r, h: sstf(r, h),
    "SCAN": lambda r, h: scan(r, h, DISK_SIZE),
    "C-SCAN": lambda r, h: cscan(r, h, DISK_SIZE),
    "LOOK": lambda r, h: look(r, h),
    "C-LOOK": lambda r, h: clook(r, h),
}

workloads = {
    "Random": lambda: random_workload(REQUEST_COUNT, DISK_SIZE),
    "Sequential": lambda: sequential_workload(
        REQUEST_COUNT, start=50
    ),
    "Localized": lambda: localized_workload(
        REQUEST_COUNT, center=100, spread=15,
        disk_size=DISK_SIZE,
    ),
    "Bursty": lambda: bursty_workload(REQUEST_COUNT, DISK_SIZE),
}

records = []
grouped = defaultdict(list)

for workload_name, generate in workloads.items():
    for trial in range(TRIALS):
        # Reproducible request trace shared by all algorithms.
        random.seed(42 + trial)
        requests = generate()

        state = environment.get_state(requests, INITIAL_HEAD)

        # Select the learned action for this workload state.
        values = q_table.get(state, {})
        rl_action = (
            max(values, key=values.get)
            if values
            else "FCFS"
        )

        # Include RL and all traditional baselines.
        for name, algorithm in algorithms.items():
            order, movement = algorithm(list(requests), INITIAL_HEAD)

            row = {
                "workload": workload_name,
                "trial": trial + 1,
                "state": state,
                "algorithm": name,
                "head_movement": movement,
                "is_rl_selected": name == rl_action,
            }
            records.append(row)
            grouped[(workload_name, name)].append(movement)

        # RL uses the same underlying implementation as its selected policy.
        _, rl_movement = algorithms[rl_action](
            list(requests), INITIAL_HEAD
        )

        row = {
            "workload": workload_name,
            "trial": trial + 1,
            "state": state,
            "algorithm": "RL",
            "head_movement": rl_movement,
            "is_rl_selected": True,
        }
        records.append(row)
        grouped[(workload_name, "RL")].append(rl_movement)

# Save detailed results.
details_path = RESULTS_DIR / "rl_scheduler_trials.csv"

with details_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print(f"\nTrials per workload: {TRIALS}")
print(f"Requests per trial: {REQUEST_COUNT}")
print(f"Disk size: {DISK_SIZE}\n")

print(
    f"{'Workload':12} | {'Algorithm':10} | "
    f"{'Avg Movement':12} | {'Std Dev':8}"
)
print("-" * 55)

summary = []

for workload_name in workloads:
    for name in [*algorithms.keys(), "RL"]:
        values = grouped[(workload_name, name)]
        avg = mean(values)
        deviation = stdev(values) if len(values) > 1 else 0

        summary.append({
            "workload": workload_name,
            "algorithm": name,
            "avg_head_movement": avg,
            "std_head_movement": deviation,
        })

        print(
            f"{workload_name:12} | {name:10} | "
            f"{avg:12.2f} | {deviation:8.2f}"
        )

summary_path = RESULTS_DIR / "rl_scheduler_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

print(f"\nDetailed results: {details_path}")
print(f"Summary results: {summary_path}")
