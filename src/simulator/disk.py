class Disk:
    def __init__(self, size=200, initial_head=53):
        self.size = size
        self.head_position = initial_head
        self.total_head_movement = 0

    def move_head(self, new_position):
        movement = abs(new_position - self.head_position)

        self.total_head_movement += movement
        self.head_position = new_position

        return movement

    def reset(self, head_position=53):
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