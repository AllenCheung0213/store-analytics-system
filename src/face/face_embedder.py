from insightface.app import FaceAnalysis

class FaceEmbedder:
    """
    Extracts a 512-d embedding vector from an aligned face.
    """
    def __init__(self):
        self.app = FaceAnalysis(allowed_modules=["recognition"])
        self.app.prepare(ctx_id=0)

    def get_embedding(self, aligned_face):
        """
        Args:
          aligned_face: 112×112 BGR image
        Returns:
          numpy.ndarray embedding
        """
        faces = self.app.get(aligned_face)
        if not faces:
            return None
        return faces[0].embedding