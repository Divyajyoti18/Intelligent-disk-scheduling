
import csv
from pathlib import Path

import random

import matplotlib.pyplot as plt

from simulator.cache import DiskCache
from simulator.predictive_cache import PredictiveCache
from simulator.workload import (
    random_workload,
    sequential_workload,
    localized_workload,
    bursty_workload
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CACHE_CAPACITY = 10
REQUEST_COUNT = 100

random.seed(42)

workloads = {
    "Random": lambda: random_workload(REQUEST_COUNT, 200),
    "Sequential": lambda: sequential_workload(REQUEST_COUNT),
    "Localized": lambda: localized_workload(REQUEST_COUNT, disk_size=200),
    "Bursty": lambda: bursty_workload(REQUEST_COUNT, 200)
}

results = []

for workload_name, generate in workloads.items():
    requests = generate()

    for cache_name, cache_class in [
        ("LRU", DiskCache),
        ("Predictive", PredictiveCache)
    ]:
        cache = cache_class(capacity=CACHE_CAPACITY)

        for position in requests:
            cache.access(position)

        stats = cache.stats()

        results.append({
            "workload": workload_name,
            "cache": cache_name,
            "hits": stats["hits"],
            "misses": stats["misses"],
            "hit_rate": stats["hit_rate"],
            "prefetches": stats.get("prefetches", 0),
            "useful_prefetches": stats.get("useful_prefetches", 0),
            "evictions": stats.get("evictions", 0)
        })

        
        print(
            f"{workload_name:12} | {cache_name:10} | "
            f"Hits: {stats['hits']:3} | "
            f"Misses: {stats['misses']:3} | "
            f"Hit rate: {stats['hit_rate']:.2%} | "
            f"Useful prefetches: {stats.get('useful_prefetches', 0)} | "
            f"Prefetches: {stats.get('prefetches', 0)}"
        )


csv_path = RESULTS_DIR / "cache_comparison.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

workload_names = list(workloads.keys())
x_positions = list(range(len(workload_names)))
bar_width = 0.35

plt.figure(figsize=(10, 6))

for index, cache_name in enumerate(["LRU", "Predictive"]):
    hit_rates = [
        next(
            row["hit_rate"]
            for row in results
            if row["workload"] == workload
            and row["cache"] == cache_name
        ) * 100
        for workload in workload_names
    ]

    offsets = [
        position + (index - 0.5) * bar_width
        for position in x_positions
    ]

    plt.bar(offsets, hit_rates, bar_width, label=cache_name)

plt.xticks(x_positions, workload_names)
plt.xlabel("Workload")
plt.ylabel("Cache Hit Rate (%)")
plt.title("LRU vs Predictive Cache")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

chart_path = FIGURES_DIR / "cache_comparison.png"
plt.savefig(chart_path, dpi=200)
plt.close()

print(f"\nCSV saved: {csv_path}")
print(f"Chart saved: {chart_path}")
