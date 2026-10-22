# ============================================================
# INTELLIGENT DISK SCHEDULING
# Traditional Disk Scheduling Algorithms
# ============================================================

# We will implement:
# 1. FCFS
# 2. SSTF
# 3. SCAN
# 4. C-SCAN
# 5. LOOK
# 6. C-LOOK


# ============================================================
# COMMON HELPER FUNCTION
# ============================================================

def calculate_movement(initial_head, path):
    """
    Calculate total head movement.

    initial_head:
        Starting position of the disk head.

    path:
        Sequence of positions visited by the head.

    Returns:
        Total distance travelled.
    """

    total = 0

    # Initially, the head is at initial_head.
    current = initial_head

    # Visit each position in the path.
    for position in path:

        # Distance between current and next position.
        movement = abs(position - current)

        # Add the movement to the total.
        total += movement

        # Update current head position.
        current = position

    return total


# ============================================================
# 1. FCFS - FIRST COME FIRST SERVE
# ============================================================

def fcfs(requests, initial_head):
    """
    Service requests in their original order.
    """

    # Copy the list so we don't modify the original.
    order = requests.copy()

    # Calculate the total head movement.
    movement = calculate_movement(initial_head, order)

    return order, movement


# ============================================================
# 2. SSTF - SHORTEST SEEK TIME FIRST
# ============================================================

def sstf(requests, initial_head):
    """
    Select the request closest to the current head position.
    """

    # Copy the pending requests.
    pending = requests.copy()

    # Store the order in which requests are serviced.
    order = []

    # Start at the initial head position.
    current = initial_head

    # Continue until every request has been serviced.
    while pending:

        # Find the request with minimum distance from the head.
        nearest = min(
            pending,
            key=lambda position: abs(position - current)
        )

        # Service the nearest request.
        order.append(nearest)

        # Update the current head position.
        current = nearest

        # Remove the serviced request.
        pending.remove(nearest)

    # Calculate total movement.
    movement = calculate_movement(initial_head, order)

    return order, movement


# ============================================================
# 3. SCAN
# ============================================================

def scan(requests, initial_head, disk_size=200, direction="right"):
    """
    The head moves in one direction, reaches the disk boundary,
    and then reverses direction.

    direction:
        "right" = move toward higher cylinder numbers first.
        "left"  = move toward lower cylinder numbers first.
    """

    if not requests:
        return [], 0

    # Separate requests on either side of the current head.
    left = sorted(
        [position for position in requests if position < initial_head]
    )

    right = sorted(
        [position for position in requests if position >= initial_head]
    )

    # This path records all positions visited by the head.
    path = []

    # This list records actual requests serviced.
    order = []

    if direction == "right":

        # Service requests to the right in ascending order.
        order.extend(right)
        path.extend(right)

        # Reach the rightmost disk boundary.
        if not path or path[-1] != disk_size - 1:
            path.append(disk_size - 1)

        # Reverse direction and service left-side requests.
        order.extend(reversed(left))
        path.extend(reversed(left))

    elif direction == "left":

        # Service requests to the left in descending order.
        order.extend(reversed(left))
        path.extend(reversed(left))

        # Reach the leftmost boundary.
        if not path or path[-1] != 0:
            path.append(0)

        # Reverse direction and service right-side requests.
        order.extend(right)
        path.extend(right)

    else:
        raise ValueError("Direction must be 'left' or 'right'")

    # Include boundary movement in the total.
    movement = calculate_movement(initial_head, path)

    return order, movement


# ============================================================
# 4. C-SCAN - CIRCULAR SCAN
# ============================================================

def cscan(requests, initial_head, disk_size=200):
    """
    Move toward higher cylinder numbers.

    After reaching the right boundary, return to cylinder 0
    and continue servicing requests in ascending order.

    The return from the right boundary to zero is counted
    in our total movement.
    """

    if not requests:
        return [], 0

    # Requests at or to the right of the current head.
    right = sorted(
        [position for position in requests if position >= initial_head]
    )

    # Requests to the left of the current head.
    left = sorted(
        [position for position in requests if position < initial_head]
    )

    # Actual request service order.
    order = right + left

    # Actual path travelled, including boundaries and wraparound.
    path = []

    # First service the right-side requests.
    path.extend(right)

    # Go to the right boundary.
    if not path or path[-1] != disk_size - 1:
        path.append(disk_size - 1)

    # Wrap around to the left boundary.
    path.append(0)

    # Service the left-side requests in ascending order.
    path.extend(left)

    # Calculate total movement, including the wraparound.
    movement = calculate_movement(initial_head, path)

    return order, movement


# ============================================================
# 5. LOOK
# ============================================================

def look(requests, initial_head, direction="right"):
    """
    Like SCAN, but the head reverses at the last request
    instead of travelling all the way to the disk boundary.
    """

    # Divide requests into left and right groups.
    left = sorted(
        [position for position in requests if position < initial_head]
    )

    right = sorted(
        [position for position in requests if position >= initial_head]
    )

    if direction == "right":

        # Service right-side requests first.
        order = right + list(reversed(left))

    elif direction == "left":

        # Service left-side requests first.
        order = list(reversed(left)) + right

    else:
        raise ValueError("Direction must be 'left' or 'right'")

    # No artificial boundary movement is required.
    movement = calculate_movement(initial_head, order)

    return order, movement


# ============================================================
# 6. C-LOOK - CIRCULAR LOOK
# ============================================================

def clook(requests, initial_head):
    """
    Service higher requests first.

    Then jump to the smallest pending request and continue
    in ascending order.

    The jump is included in the movement calculation.
    """

    # Requests at or to the right of the head.
    right = sorted(
        [position for position in requests if position >= initial_head]
    )

    # Requests to the left of the head.
    left = sorted(
        [position for position in requests if position < initial_head]
    )

    # Service right-side requests, then wrap to the smallest
    # left-side request and continue in ascending order.
    order = right + left

    # Calculate movement, including the wraparound distance.
    movement = calculate_movement(initial_head, order)

    return order, movement
