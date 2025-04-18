from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, MetaData, Table

engine = create_engine("sqlite:///analytics.db")
metadata = MetaData()
events = Table("events", metadata,
  Column("id", Integer, primary_key=True),
  Column("timestamp", DateTime), Column("person_id", Integer),
  Column("is_repeat", Boolean), Column("age_group", String),
  Column("gender", String)
)
metadata.create_all(engine)
def insert_event(rec):
  engine.execute(events.insert(), rec)
