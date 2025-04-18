class EntryCounter:
    """
    Counts entries by detecting when track centroids cross a virtual line.
    """
    def __init__(self, line_y):
        self.line_y = line_y
        self.history = {}  # track_id -> last centroid y-coordinate

    def update(self, tracks):
        """
        Args:
          tracks: dict of track_id -> [x1,y1,x2,y2]
        Returns:
          list of track_id that just crossed the line (new entries)
        """
        entries = []
        for tid, bbox in tracks.items():
            x1, y1, x2, y2 = bbox
            cy = (y1 + y2) // 2
            prev = self.history.get(tid, cy)
            # if moved from above line to below line
            if prev < self.line_y <= cy:
                entries.append(tid)
            self.history[tid] = cy
        return entries