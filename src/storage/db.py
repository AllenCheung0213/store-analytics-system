from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, MetaData, Table
import datetime

# SQLite local storage
engine = create_engine("sqlite:///analytics.db", echo=False)
metadata = MetaData()

events = Table(
    "events", metadata,
    Column("id", Integer, primary_key=True),
    Column("timestamp", DateTime, default=datetime.datetime.utcnow),
    Column("person_id", Integer, nullable=False),
    Column("is_repeat", Boolean, nullable=False),
    Column("age_group", String),
    Column("gender", String),
)
metadata.create_all(engine)

def insert_event(person_id, is_repeat, age_group, gender):
    """
    Inserts a single event record into the local analytics database.
    """
    ins = events.insert().values(
        timestamp=datetime.datetime.utcnow(),
        person_id=person_id,
        is_repeat=is_repeat,
        age_group=age_group,
        gender=gender
    )
    conn = engine.connect()
    conn.execute(ins)
    conn.close()
