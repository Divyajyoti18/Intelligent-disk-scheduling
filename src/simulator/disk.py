class Disk:
    def __init__(self, size=200, initial_head=53):
        if size <= 0:
            raise ValueError("Disk size must be positive")
        if not 0 <= initial_head < size:
            raise ValueError("Initial head position must be within the disk")

        self.size = size
        self.initial_head = initial_head
        self.head_position = initial_head
        self.total_head_movement = 0

    def move_head(self, new_position):
        if not 0 <= new_position < self.size:
            raise ValueError("Head position must be within the disk")

        movement = abs(new_position - self.head_position)

        self.total_head_movement += movement
        self.head_position = new_position

        return movement

    def reset(self, head_position=None):
        if head_position is None:
            head_position = self.initial_head
        if not 0 <= head_position < self.size:
            raise ValueError("Head position must be within the disk")

        self.head_position = head_position
        self.total_head_movement = 0

    def __repr__(self):
        return (
            f"Disk("
            f"size={self.size}, "
            f"head={self.head_position}, "
            f"movement={self.total_head_movement}"
            f")"
        )
