from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    """
    Wraps Deep SORT for multi-object tracking.
    """
    def __init__(self, max_age=30):
        # max_age: frames to keep lost tracks
        self.tracker = DeepSort(max_age=max_age)

    def update(self, frame, detections):
        """
        Args:
          frame: current BGR image
          detections: list of [x1,y1,x2,y2] ints
        Returns:
          dict of track_id -> [x1,y1,x2,y2]
        """
        # DeepSort expects (xmin,ymin,xmax,ymax,confidence)
        det_list = [(x1, y1, x2, y2, 1.0) for x1, y1, x2, y2 in detections]
        tracks = self.tracker.update_tracks(det_list, frame=frame)
        output = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
            output[track_id] = [int(x1), int(y1), int(x2), int(y2)]
        return output