class DiskRequest:
    def __init__(self, request_id, position, arrival_time, request_type="READ"):
        self.request_id = request_id
        self.position = position
        self.arrival_time = arrival_time
        self.request_type = request_type

    def __repr__(self):
        return (
            f"DiskRequest("
            f"id={self.request_id}, "
            f"position={self.position}, "
            f"arrival={self.arrival_time}, "
            f"type={self.request_type}"
            f")"
        )