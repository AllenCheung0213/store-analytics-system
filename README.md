# Foot Traffic Analytics Pipeline

This repository implements a real‑time foot‑traffic and customer analytics system for a retail environment using CCTV video feeds. The pipeline detects and counts people entering a store, performs face re‑identification to track repeat visitors, estimates demographic attributes (age and gender), and stores anonymized analytics for downstream insights.

## Features

- **Real‑time Person Detection**: Uses YOLOv5/YOLOv8 (Ultralytics) to detect people in each frame.
- **Multi‑Object Tracking**: Integrates Deep SORT to maintain consistent track IDs across frames.
- **Entry Counting**: Defines a virtual entry line; counts a new entry when a person’s track crosses the line.
- **Face Processing**:
  - *Detection & Alignment*: RetinaFace via InsightFace to detect faces and landmarks; aligns faces to a fixed size.
  - *Embedding & Re‑Identification*: ArcFace embeddings for each face; nearest‑neighbor lookup to assign persistent visitor IDs.
  - *Demographics*: Age and gender prediction using OpenCV Caffe models (Adience dataset).
- **Local Storage**: SQLite database (`analytics.db`) to store events (`timestamp`, `person_id`, `is_repeat`, `age_group`, `gender`).
- **Modular Design**: Clean separation between video ingestion, detection, tracking, face processing, and storage modules.
- **Testing**: Pytest suite covering core logic (entry counting, re‑ID, demographics, database operations).

## Repository Structure

```
foot_traffic_pipeline/
├── configs/
│   └── camera.yaml          # Camera stream and processing settings
├── models/                  # Pre-trained face detection & age/gender models
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
├── src/
│   ├── video/
│   │   └── frame_grabber.py
│   ├── events/
│   │   └── event_builder.py
│   ├── detection/
│   │   └── person_detector.py
│   ├── tracking/
│   │   ├── tracker.py
│   │   └── entry_counter.py
│   ├── face/
│   │   ├── face_detector.py
│   │   ├── face_embedder.py
│   │   ├── reid.py
│   │   └── demographics.py
│   ├── storage/
│   │   └── db.py
│   └── main.py             # Orchestrates entire pipeline
├── tests/
│   └── test_foot_traffic.py # Pytest suite
├── analytics.db            # SQLite database (auto-created)
├── README.md
└── requirements.txt        # Pinned Python dependencies
```

## Configuration

1. **`configs/camera.yaml`**
   ```yaml
   rtsp_url: "rtsp://<username>:<password>@<camera_ip>:554/your_stream"
   target_fps: 5
   frame_width: 1280
   frame_height: 720
   ```
2. **Models**: Place the pre-trained Caffe files for age/gender under `models/`.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/AllenCheung0213/store-analytics-system.git
   cd store-analytics-system
   ```
2. Create a Python virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

1. **Run the pipeline**:
   ```bash
   python src/main.py
   ```
2. **Monitor**: A window will display live video with detection overlays (press `Esc` to exit).
3. **Data**: Events are logged into `analytics.db` (SQLite). Use any SQLite client to query:
   ```sql
   SELECT * FROM events ORDER BY timestamp DESC LIMIT 10;
   ```

## Testing

Execute the test suite with pytest:
```bash
pytest --maxfail=1 --disable-warnings -q
```

## Extending & Deployment

- **Adjust thresholds**: Tune re‑ID `threshold` in `src/face/reid.py` or entry line in `EntryCounter`.
- **Swapping models**: Replace `yolov5n.pt` with a larger or custom YOLO model; adjust face models as needed.
- **Cloud sync**: Implement the optional FastAPI sync service to push events to a central database.
- **Dockerization**: Add `Dockerfile` and `docker-compose.yml` to containerize the pipeline and optional API.

## License

This project is licensed under the [Apache-2.0 license](LICENSE).
