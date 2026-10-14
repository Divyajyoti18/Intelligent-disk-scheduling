# ============================================================
# MAIN PROGRAM
# Compare traditional disk scheduling algorithms
# ============================================================

# Import all six scheduling algorithms.
from schedulers.algorithms import (
    fcfs,
    sstf,
    scan,
    cscan,
    look,
    clook
)


# ============================================================
# INPUT DATA
# ============================================================

# Total number of disk cylinders: 0 to 199.
DISK_SIZE = 200

# Initial position of the disk head.
INITIAL_HEAD = 53

# Disk requests.
REQUESTS = [98, 183, 37, 122, 14, 124, 65, 67]


# ============================================================
# RUN ALL ALGORITHMS
# ============================================================

# Store each algorithm's name and corresponding function.
algorithms = [
    ("FCFS", lambda: fcfs(REQUESTS, INITIAL_HEAD)),

    ("SSTF", lambda: sstf(REQUESTS, INITIAL_HEAD)),

    ("SCAN", lambda: scan(
        REQUESTS, INITIAL_HEAD, DISK_SIZE, "right"
    )),

    ("C-SCAN", lambda: cscan(
        REQUESTS, INITIAL_HEAD, DISK_SIZE
    )),

    ("LOOK", lambda: look(
        REQUESTS, INITIAL_HEAD, "right"
    )),

    ("C-LOOK", lambda: clook(
        REQUESTS, INITIAL_HEAD
    ))
]


# Store results for display and later analysis.
results = []


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 65)
print("INTELLIGENT DISK SCHEDULING - BASELINE COMPARISON")
print("=" * 65)

print(f"Disk size       : {DISK_SIZE} cylinders")
print(f"Initial head    : {INITIAL_HEAD}")
print(f"Requests        : {REQUESTS}")

print("\n" + "-" * 65)


# Run each scheduling algorithm.
for name, algorithm in algorithms:

    # Each function returns:
    # 1. Service order
    # 2. Total head movement
    order, movement = algorithm()

    # Save the result.
    results.append((name, order, movement))

    # Display the result.
    print(f"\nAlgorithm       : {name}")
    print(f"Service order   : {order}")
    print(f"Head movement   : {movement} cylinders")


# ============================================================
# FIND THE LOWEST HEAD MOVEMENT
# ============================================================

best_algorithm = min(
    results,
    key=lambda result: result[2]
)

print("\n" + "=" * 65)
print("BASELINE SUMMARY")
print("=" * 65)

for name, order, movement in results:
    print(f"{name:<10} : {movement:>4} cylinders")

print("\nLowest head movement:")
print(f"{best_algorithm[0]} : {best_algorithm[2]} cylinders")

print("=" * 65)