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

workloads = {
    "Random": lambda: random_workload(REQUEST_COUNT, DISK_SIZE),
    "Sequential": lambda: sequential_workload(REQUEST_COUNT),
    "Localized": lambda: localized_workload(
        REQUEST_COUNT, disk_size=DISK_SIZE
    ),
    "Bursty": lambda: bursty_workload(REQUEST_COUNT, DISK_SIZE)
}

algorithms = {
    "FCFS": lambda requests: fcfs(requests, INITIAL_HEAD),
    "SSTF": lambda requests: sstf(requests, INITIAL_HEAD),
    "SCAN": lambda requests: scan(
        requests, INITIAL_HEAD, DISK_SIZE, "right"
    ),
    "C-SCAN": lambda requests: cscan(
        requests, INITIAL_HEAD, DISK_SIZE
    ),
    "LOOK": lambda requests: look(
        requests, INITIAL_HEAD, "right"
    ),
    "C-LOOK": lambda requests: clook(requests, INITIAL_HEAD)
}

all_results = []

for workload_name, generate in workloads.items():
    requests = generate()

    print(f"\n{'=' * 65}")
    print(f"WORKLOAD: {workload_name}")
    print(f"Requests: {requests}")
    print("-" * 65)
    print(f"{'Algorithm':<12}{'Head Movement':>18}")

    for algorithm_name, run_algorithm in algorithms.items():
        order, movement = run_algorithm(requests)
        all_results.append(
            (workload_name, algorithm_name, movement)
        )

        print(f"{algorithm_name:<12}{movement:>18}")

print(f"\n{'=' * 65}")
print("EXPERIMENTS COMPLETED")
print(f"Workloads tested: {len(workloads)}")
print(f"Algorithms tested: {len(algorithms)}")
print(f"Total comparisons: {len(all_results)}")