from insightface.app import FaceAnalysis
import numpy as np

class FaceDetector:
    """
    Detects faces and facial landmarks using InsightFace (RetinaFace).
    """
    def __init__(self):
        self.app = FaceAnalysis(allowed_modules=["detection"])
        # ctx_id=0 for GPU, ctx_id=-1 for CPU
        self.app.prepare(ctx_id=0, det_size=(512, 512))

    def detect(self, image):
        """
        Args:
          image: BGR numpy array
        Returns:
          list of Face objects with .bbox and .kps (landmarks)
        """
        faces = self.app.get(image)
        return faces
    
def align_face(image, face):
    """
    Aligns and crops the face to 112×112 using 5 landmarks.
    """
    import cv2
    src = np.array(face.kps, dtype=np.float32)
    dst = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]
    ], dtype=np.float32)
    tform, _ = cv2.estimateAffinePartial2D(src, dst)
    aligned = cv2.warpAffine(image, tform, (112, 112))
    return aligned