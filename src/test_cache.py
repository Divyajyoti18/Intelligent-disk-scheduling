def main():

    from simulator.cache import DiskCache

    cache = DiskCache(capacity=3)
    requests = [10, 20, 10, 30, 40, 20, 40]

    for position in requests:
        hit = cache.access(position)
        print(f"Position {position}: {'HIT' if hit else 'MISS'}")

    print("\nCache statistics:")
    print(cache.stats())


if __name__ == '__main__':
    main()
