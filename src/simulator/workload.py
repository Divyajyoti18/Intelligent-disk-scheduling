import random


def random_workload(count, disk_size=200):
    return [random.randint(0, disk_size - 1) for _ in range(count)]



import random


def sequential_workload(request_count=100, disk_size=200):
    """Generate consecutive requests that stay within disk boundaries."""
    if disk_size <= 0:
        raise ValueError("Disk size must be positive")

    if request_count < 0:
        raise ValueError("Request count cannot be negative")

    if request_count == 0:
        return []

    # Wrap around the disk if more requests than disk positions
    # are required.
    start = random.randint(0, disk_size - 1)

    return [
        (start + i) % disk_size
        for i in range(request_count)
    ]



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
