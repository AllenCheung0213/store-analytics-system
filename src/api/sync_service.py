from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import psycopg2
import datetime

class Event(BaseModel):
    timestamp: datetime.datetime
    person_id: int
    is_repeat: bool
    age_group: str
    gender: str

app = FastAPI()

@app.post("/ingest")
def ingest(events: List[Event]):
    # conn = psycopg2.connect(...central_db_uri...)
    cur = conn.cursor()
    for e in events:
        cur.execute(
          "INSERT INTO events(...) VALUES (%s,%s,%s,%s,%s)",
          (e.timestamp,e.person_id,e.is_repeat,e.age_group,e.gender)
        )
    conn.commit(); cur.close(); conn.close()
    return {"status":"ok", "count": len(events)}
