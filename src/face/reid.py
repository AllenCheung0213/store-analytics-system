import numpy as np

class ReID:
    """
    Simple nearest-neighbor face re-identification in embedding space.
    """
    def __init__(self, threshold=1.0):
        self.db = {}          # person_id -> embedding vector
        self.threshold = threshold
        self.next_id = 1

    def lookup(self, embedding):
        """
        Args:
          embedding: numpy.ndarray
        Returns:
          (person_id, is_repeat)
        """
        if embedding is None:
            return None, False
        if not self.db:
            pid = self.next_id
            self.db[pid] = embedding
            self.next_id += 1
            return pid, False
        # compute distances
        dists = {pid: np.linalg.norm(embedding - emb)
                 for pid, emb in self.db.items()}
        best_pid, best_dist = min(dists.items(), key=lambda x: x[1])
        if best_dist < self.threshold:
            return best_pid, True
        pid = self.next_id
        self.db[pid] = embedding
        self.next_id += 1
        return pid, False