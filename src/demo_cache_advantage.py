
from collections import OrderedDict

requests = [10, 20, 30] * 10
capacity = 2


def lru_hit_rate(requests, capacity):
    cache = OrderedDict()
    hits = 0

    for position in requests:
        if position in cache:
            hits += 1
            cache.move_to_end(position)
        else:
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[position] = True

    return hits, len(requests) - hits


def ideal_prefetch_hit_rate(requests, capacity):
    cache = set()
    hits = 0
    misses = 0

    for i, position in enumerate(requests):
        if position in cache:
            hits += 1
        else:
            misses += 1

        if i + 1 < len(requests):
            next_position = requests[i + 1]

            if next_position not in cache:
                if len(cache) >= capacity:
                    future = requests[i + 2:]
                    victims = [
                        item for item in cache
                        if item not in future
                    ]

                    if victims:
                        cache.remove(victims[0])
                    else:
                        cache.pop()

                cache.add(next_position)

        if position not in cache and len(cache) < capacity:
            cache.add(position)

    return hits, misses


for name, method in [
    ("LRU", lru_hit_rate),
    ("Idealized predictive upper bound", ideal_prefetch_hit_rate)
]:
    hits, misses = method(requests, capacity)
    print(f"{name}: hits={hits}, misses={misses}, "
          f"hit rate={hits / len(requests):.2%}")



import matplotlib.pyplot as plt

# Results from the idealized demonstration
algorithms = ["LRU Baseline", "Idealized Predictive"]
hit_rates = [0.00, 96.67]
misses = [30, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Graph 1: Cache hit rate
axes[0].bar(algorithms, hit_rates)
axes[0].set_title("Cache Hit Rate Comparison")
axes[0].set_ylabel("Hit Rate (%)")
axes[0].set_ylim(0, 110)
axes[0].bar_label(
    axes[0].containers[0],
    fmt="%.2f%%"
)

# Graph 2: Cache misses
axes[1].bar(algorithms, misses)
axes[1].set_title("Cache Miss Comparison")
axes[1].set_ylabel("Number of Misses")
axes[1].bar_label(axes[1].containers[0])

plt.suptitle("LRU vs Idealized Predictive Caching")
plt.tight_layout()

plt.savefig(
    "results/figures/demo_cache_advantage.png",
    dpi=200,
    bbox_inches="tight"
)
plt.show()
