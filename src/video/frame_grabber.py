import cv2
import yaml

class FrameGrabber:
    """
    Captures frames from an RTSP camera stream at a configured FPS and resolution.
    Usage:
        grabber = FrameGrabber()
        for frame in grabber:
            # process frame
    """
    def __init__(self, config_path="configs/camera.yaml"):
        cfg = yaml.safe_load(open(config_path))
        self.cap = cv2.VideoCapture(cfg["rtsp_url"])
        # enforce target FPS and resolution
        self.cap.set(cv2.CAP_PROP_FPS, cfg.get("target_fps", 5))
        if cfg.get("frame_width"):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["frame_width"])
        if cfg.get("frame_height"):
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["frame_height"])
        # compute delay for display
        self.delay = int(1000 / cfg.get("target_fps", 5))

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def show(self):
        """Utility to display live feed for debug"""
        for frame in self:
            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(self.delay) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
