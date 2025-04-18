import pytest
import numpy as np
import cv2
import datetime
from sqlalchemy import create_engine, text

# Import the modules to test
from src.tracking.entry_counter import EntryCounter
from src.face.reid import ReID
from src.face.demographics import predict_age_gender, AGE_BUCKETS, GENDERS
from src.storage.db import insert_event, engine, events, metadata


def test_entry_counter_cross():
    # Initialize with a horizontal line at y=150
    ec = EntryCounter(line_y=150)
    # First frame: centroid below line? No entry
    tracks_frame1 = {1: [0, 100, 10, 140]}  # centroid y = 120
    entries1 = ec.update(tracks_frame1)
    assert entries1 == []

    # Second frame: centroid now crosses below line
    tracks_frame2 = {1: [0, 100, 10, 160]}  # centroid y = 130? Actually: (100+160)//2 = 130
    # adjust to cross: set y2 to 200 => centroid = 150
    tracks_frame2[1] = [0, 100, 10, 200]
    entries2 = ec.update(tracks_frame2)
    assert entries2 == [1]

    # Additional update shouldn't recount
    entries3 = ec.update(tracks_frame2)
    assert entries3 == []


def test_reid_new_and_repeat():
    # Use a small threshold for clarity
    reid = ReID(threshold=0.5)
    emb1 = np.zeros((512,))
    pid1, is_rep1 = reid.lookup(emb1)
    assert pid1 == 1
    assert not is_rep1

    # Slightly different embedding within threshold => repeat
    emb1_mod = np.zeros((512,)) + 0.1
    pid2, is_rep2 = reid.lookup(emb1_mod)
    assert pid2 == pid1
    assert is_rep2

    # Far embedding => new ID
    emb2 = np.ones((512,)) * 10
    pid3, is_rep3 = reid.lookup(emb2)
    assert pid3 != pid1
    assert not is_rep3


def test_predict_age_gender_valid_labels():
    # Create a dummy face image of correct size for OpenCV model
    dummy_face = np.zeros((227, 227, 3), dtype=np.uint8)
    age, gender = predict_age_gender(dummy_face)
    assert isinstance(age, str)
    assert isinstance(gender, str)
    assert age in AGE_BUCKETS
    assert gender in GENDERS


def test_db_insert_and_query(tmp_path, monkeypatch):
    # Create a fresh in-memory SQLite for testing
    test_engine = create_engine("sqlite:///:memory:")
    # Recreate schema
    metadata.create_all(test_engine)

    # Monkeypatch the engine in storage.db
    import src.storage.db as db_mod
    monkeypatch.setattr(db_mod, 'engine', test_engine)
    monkeypatch.setattr(db_mod, 'events', events)

    # Insert an event
    insert_event(person_id=42, is_repeat=True, age_group="25-32", gender="Female")

    # Query the test DB directly
    with test_engine.connect() as conn:
        result = conn.execute(text("SELECT person_id, is_repeat, age_group, gender FROM events")).fetchall()
    assert len(result) == 1
    row = result[0]
    assert row['person_id'] == 42
    assert row['is_repeat'] == 1 or row['is_repeat'] is True
    assert row['age_group'] == "25-32"
    assert row['gender'] == "Female"
