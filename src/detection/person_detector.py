from ultralytics import YOLO
import numpy as np

class PersonDetector:
    """
    Detects persons in a frame using YOLOv5/YOLOv8.
    """
    def __init__(self, model_path="yolov5n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Returns a list of bounding boxes [[x1,y1,x2,y2], ...]
        only for class 'person'.
        """
        results = self.model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # shape (N,4)
        return boxes.astype(int).tolist()