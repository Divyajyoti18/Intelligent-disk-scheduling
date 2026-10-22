def main():

    import csv
    import random
    import statistics
    from collections import defaultdict
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from simulator.cache import DiskCache
    from simulator.predictive_cache import PredictiveCache
    from simulator.workload import (
        random_workload,
        sequential_workload,
        localized_workload,
        bursty_workload,
    )


    ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = ROOT / "results"
    FIGURES_DIR = RESULTS_DIR / "figures"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    CACHE_CAPACITY = 10
    REQUEST_COUNT = 100
    TRIALS = 30
    BASE_SEED = 42

    workloads = {
        "Random": lambda: random_workload(REQUEST_COUNT, 200),
        "Sequential": lambda: sequential_workload(
            REQUEST_COUNT, disk_size=200
        ),
        "Localized": lambda: localized_workload(
            REQUEST_COUNT, disk_size=200
        ),
        "Bursty": lambda: bursty_workload(REQUEST_COUNT, 200),
    }

    detailed_results = []
    grouped_results = defaultdict(list)

    # Each trial uses one trace shared by both algorithms.
    for workload_name, generate in workloads.items():
        for trial in range(TRIALS):
            seed = BASE_SEED + trial

            random.seed(seed)
            requests = generate()

            for cache_name, cache_class in [
                ("LRU", DiskCache),
                ("Predictive", PredictiveCache),
            ]:
                cache = cache_class(capacity=CACHE_CAPACITY)

                for position in requests:
                    cache.access(position)

                stats = cache.stats()

                row = {
                    "trial": trial + 1,
                    "seed": seed,
                    "workload": workload_name,
                    "cache": cache_name,
                    "hits": stats["hits"],
                    "misses": stats["misses"],
                    "hit_rate": stats["hit_rate"],
                    "evictions": stats.get("evictions", 0),
                    "prefetches": stats.get("prefetches", 0),
                    "useful_prefetches": stats.get(
                        "useful_prefetches", 0
                    ),
                    "wasted_prefetches": stats.get(
                        "wasted_prefetches", 0
                    ),
                "demand_evictions_for_prefetches": stats.get(
                    "demand_evictions_for_prefetches", 0
                ),
                    "prediction_attempts": stats.get("prediction_attempts", 0),
                    "correct_predictions": stats.get("correct_predictions", 0),
                    "prediction_accuracy": stats.get("prediction_accuracy", 0),
                }

                detailed_results.append(row)
                grouped_results[(workload_name, cache_name)].append(row)

    # Save every individual trial.
    details_path = RESULTS_DIR / "cache_trials.csv"

    with details_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=detailed_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(detailed_results)

    # Calculate means and standard deviations.
    summary_results = []

    for (workload_name, cache_name), rows in grouped_results.items():
        summary = {
            "workload": workload_name,
            "cache": cache_name,
            "trials": len(rows),
        }

        for metric in [
            "hits",
            "misses",
            "hit_rate",
            "evictions",
            "prefetches",
            "useful_prefetches",
            "wasted_prefetches",
        "demand_evictions_for_prefetches",
            "prediction_attempts",
            "correct_predictions",
            "prediction_accuracy",
        ]:
            values = [row[metric] for row in rows]

            summary[f"avg_{metric}"] = statistics.mean(values)
            summary[f"std_{metric}"] = (
                statistics.stdev(values) if len(values) > 1 else 0
            )

        summary_results.append(summary)

    summary_path = RESULTS_DIR / "cache_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=summary_results[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary_results)

    # Print average results.
    print(f"\nTrials per workload: {TRIALS}")
    print(f"Requests per trial: {REQUEST_COUNT}")
    print(f"Cache capacity: {CACHE_CAPACITY}\n")

    print(
        f"{'Workload':12} | {'Cache':10} | {'Hit Rate':10} | "
        f"{'Useful PF':9} | {'Wasted PF':9} | "
        f"{'Pred Accuracy':13}"
    )

    for row in summary_results:
        print(
            f"{row['workload']:12} | "
            f"{row['cache']:10} | "
            f"{row['avg_hit_rate']:9.2%} | "
            f"{row['avg_useful_prefetches']:9.2f} | "
            f"{row['avg_wasted_prefetches']:9.2f} | "
            f"{row.get('avg_prediction_accuracy', 0):12.2%}"
        )
    # Plot mean hit rates with standard-deviation error bars.
    workload_names = list(workloads.keys())
    x_positions = list(range(len(workload_names)))
    bar_width = 0.35

    plt.figure(figsize=(10, 6))

    for index, cache_name in enumerate(["LRU", "Predictive"]):
        rows_by_workload = {
            row["workload"]: row
            for row in summary_results
            if row["cache"] == cache_name
        }

        means = [
            rows_by_workload[name]["avg_hit_rate"] * 100
            for name in workload_names
        ]

        errors = [
            rows_by_workload[name]["std_hit_rate"] * 100
            for name in workload_names
        ]

        offsets = [
            position + (index - 0.5) * bar_width
            for position in x_positions
        ]

        plt.bar(
            offsets,
            means,
            bar_width,
            yerr=errors,
            capsize=4,
            label=cache_name,
        )

    plt.xticks(x_positions, workload_names)
    plt.xlabel("Workload")
    plt.ylabel("Average Cache Hit Rate (%)")
    plt.title(f"LRU vs Predictive Cache ({TRIALS} Trials)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    chart_path = FIGURES_DIR / "cache_comparison_average.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print(f"\nDetailed trials: {details_path}")
    print(f"Average summary: {summary_path}")
    print(f"Average comparison chart: {chart_path}")


if __name__ == '__main__':
    main()
