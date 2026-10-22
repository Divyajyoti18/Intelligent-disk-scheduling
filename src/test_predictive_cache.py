def main():

    from simulator.predictive_cache import PredictiveCache

    cache = PredictiveCache(capacity=2)

    requests = [10, 20, 30, 40, 10, 20, 30, 40, 10, 20]

    for position in requests:
        hit = cache.access(position)
        print(f"Position {position}: {'HIT' if hit else 'MISS'}")

    print("\nPredictive cache statistics:")
    print(cache.stats())


if __name__ == '__main__':
    main()
