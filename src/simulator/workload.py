import random


def random_workload(count, disk_size=200):
    return [random.randint(0, disk_size - 1) for _ in range(count)]


def sequential_workload(count, start=50):
    return [(start + i) % 200 for i in range(count)]


def localized_workload(count, center=100, spread=15, disk_size=200):
    low = max(0, center - spread)
    high = min(disk_size - 1, center + spread)
    return [random.randint(low, high) for _ in range(count)]


def bursty_workload(count, disk_size=200):
    requests = []
    centers = [30, 100, 170]

    for i in range(count):
        center = centers[(i // 5) % len(centers)]
        position = random.randint(
            max(0, center - 10),
            min(disk_size - 1, center + 10)
        )
        requests.append(position)

    return requests