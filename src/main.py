
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt

from schedulers.algorithms import (
    fcfs, sstf, scan, cscan, look, clook
)
from simulator.workload import (
    random_workload,
    sequential_workload,
    localized_workload,
    bursty_workload
)

DISK_SIZE = 200
INITIAL_HEAD = 53
REQUEST_COUNT = 50
random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CSV_FILE = RESULTS_DIR / "baseline_results.csv"
CHART_FILE = FIGURES_DIR / "head_movement.png"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

workloads = {
    "Random": lambda: random_workload(REQUEST_COUNT, DISK_SIZE),
    "Sequential": lambda: sequential_workload(REQUEST_COUNT),
    "Localized": lambda: localized_workload(
        REQUEST_COUNT, disk_size=DISK_SIZE
    ),
    "Bursty": lambda: bursty_workload(REQUEST_COUNT, DISK_SIZE)
}

algorithms = {
    "FCFS": lambda r: fcfs(r, INITIAL_HEAD),
    "SSTF": lambda r: sstf(r, INITIAL_HEAD),
    "SCAN": lambda r: scan(r, INITIAL_HEAD, DISK_SIZE, "right"),
    "C-SCAN": lambda r: cscan(r, INITIAL_HEAD, DISK_SIZE),
    "LOOK": lambda r: look(r, INITIAL_HEAD, "right"),
    "C-LOOK": lambda r: clook(r, INITIAL_HEAD)
}

results = []

for workload_name, generate in workloads.items():
    requests = generate()

    print(f"\n{'=' * 60}")
    print(f"WORKLOAD: {workload_name}")
    print(f"{'Algorithm':<12}{'Head Movement':>18}")

    for algorithm_name, run_algorithm in algorithms.items():
        order, movement = run_algorithm(requests)

        results.append({
            "workload": workload_name,
            "algorithm": algorithm_name,
            "head_movement": movement
        })

        print(f"{algorithm_name:<12}{movement:>18}")

with CSV_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["workload", "algorithm", "head_movement"]
    )
    writer.writeheader()
    writer.writerows(results)

algorithm_names = list(algorithms.keys())
workload_names = list(workloads.keys())

x_positions = list(range(len(algorithm_names)))
bar_width = 0.18

plt.figure(figsize=(12, 6))

for index, workload_name in enumerate(workload_names):
    movements = [
        next(
            row["head_movement"]
            for row in results
            if row["workload"] == workload_name
            and row["algorithm"] == algorithm
        )
        for algorithm in algorithm_names
    ]

    offsets = [
        position + (index - 1.5) * bar_width
        for position in x_positions
    ]

    plt.bar(offsets, movements, bar_width, label=workload_name)

plt.xticks(x_positions, algorithm_names)
plt.xlabel("Disk Scheduling Algorithm")
plt.ylabel("Total Head Movement (cylinders)")
plt.title("Disk Scheduling Performance Across Workloads")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(CHART_FILE, dpi=200)
plt.close()

print(f"\nCSV saved: {CSV_FILE}")
print(f"Chart saved: {CHART_FILE}")
print(f"Total comparisons: {len(results)}")
