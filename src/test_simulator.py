def main():
    # Import the Disk class
    from simulator.disk import Disk

    # Import the DiskRequest class
    from simulator.request import DiskRequest


    # Create a disk with:
    # - 200 positions
    # - head starting at position 53

    disk = Disk(
        size=200,
        initial_head=53
    )


    # Create one disk request.
    request = DiskRequest(
        request_id=1,
        position=98,
        arrival_time=0,
        request_type="READ"
    )


    # Print the request
    print("Request:")
    print(request)


    # Move the disk head to the request position
    movement = disk.move_head(request.position)


    # Print how far the head moved
    print("\nDisk head movement:")
    print(movement)


    # Print the current state of the disk
    print("\nCurrent disk state:")
    print(disk)

if __name__ == '__main__':
    main()
