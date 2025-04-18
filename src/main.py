import cv2
from src.video.frame_grabber import FrameGrabber
from src.detection.person_detector import PersonDetector
from src.tracking.tracker import Tracker
from src.tracking.entry_counter import EntryCounter
from src.face.face_detector import FaceDetector, align_face
from src.face.face_embedder import FaceEmbedder
from src.face.reid import ReID
from src.face.demographics import predict_age_gender
from src.storage.db import insert_event


def run_pipeline():
    """
    Main loop: capture frames, detect persons, track, detect faces,
    embed & re-identify, predict demographics, and store events.
    """
    grabber = FrameGrabber()
    detector = PersonDetector()
    tracker = Tracker()
    counter = EntryCounter(line_y=400)
    face_detector = FaceDetector()
    embedder = FaceEmbedder()
    reid_db = ReID(threshold=1.0)

    for frame in grabber:
        # 1. Detect persons
        person_boxes = detector.detect(frame)
        # 2. Track
        tracks = tracker.update(frame, person_boxes)
        # 3. Count new entries
        new_ids = counter.update(tracks)
        for track_id in new_ids:
            # crop person region for face
            x1, y1, x2, y2 = tracks[track_id]
            person_img = frame[y1:y2, x1:x2]
            # 4. Face detect & align
            faces = face_detector.detect(person_img)
            if not faces:
                continue
            aligned = align_face(person_img, faces[0])
            # 5. Embedding & re-ID
            emb = embedder.get_embedding(aligned)
            person_id, is_repeat = reid_db.lookup(emb)
            # 6. Demographic prediction
            age_group, gender = predict_age_gender(aligned)
            # 7. Store event
            insert_event(person_id, is_repeat, age_group, gender)

        # Optional: display for monitoring with blurred faces
        cv2.imshow("Analytics", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()