import sqlite3
from datetime import datetime
import json
from pprint import pprint

conn = sqlite3.connect(":memory:")
conn.row_factory = sqlite3.Row
conn.execute("""PRAGMA foreign_keys = ON""")


from orchestrator.domain import Segment, SegmentVersion, SegmentType

def segment_to_row(seg: Segment) -> tuple:
     return (
        seg.id, seg.type.value, seg.name,
        seg.created_at.isoformat(),
        seg.latest_version_no
     )

def row_to_segment(row: sqlite3.Row) -> Segment:
     return Segment(
        id=row["id"], type=row["type"],
        name=row["name"],
        created_at=datetime.fromisoformat(row["created_at"]),
        latest_version_no=row["latest_version_no"]
     )

def version_to_row(v: SegmentVersion) -> tuple:
     return (
          v.segment_id, v.version_no,
          v.content, v.role_hint,
          v.token_estimate,
          json.dumps(list(v.derived_from)) if v.derived_from else None,
          v.created_at.isoformat()
     )

def row_to_version(row: sqlite3.Row) -> SegmentVersion:
     return SegmentVersion(
          segment_id=row["segment_id"],
          version_no=row["version_no"],
          content=row["content"],
          role_hint=row["role_hint"],
          token_estimate=row["token_estimate"],
          derived_from=tuple(json.loads(row["derived_from"])) if \
             row["derived_from"] else None,
          created_at=datetime.fromisoformat(row["created_at"])
     )

def append_version(conn, segment_id, version_no,
                   content, role_hint="user"):
        with conn:
            entry = conn.execute("""
                SELECT latest_version_no FROM segment WHERE id = ?
                """, (segment_id, )).fetchone()
            if version_no != entry["latest_version_no"] + 1:
                raise ValueError(f"{version_no} <> {entry['latest_version_no']} + 1")
            conn.execute("""
                INSERT INTO segment_version
                (segment_id, version_no, content, role_hint)
                VALUES (?, ?, ?, ?)""",
                (segment_id, version_no, content, role_hint ))
            conn.execute("""
                UPDATE segment SET latest_version_no = ? WHERE id = ?
                """,
                (version_no, segment_id))

conn.execute(
    """
    CREATE TABLE segment (
        id                  TEXT PRIMARY KEY,
        type                TEXT,
        name                TEXT,
        created_at          TEXT,
        latest_version_no   INTEGER
    )
    """
)

conn.execute(
    """
    CREATE TABLE segment_version (
        segment_id          TEXT,
        version_no          INTEGER NOT NULL,
        content             TEXT,
        role_hint           TEXT,
        token_estimate      INTEGER,
        derived_from        TEXT,
        created_at          TEXT,
        PRIMARY KEY (segment_id, version_no),
        FOREIGN KEY (segment_id) REFERENCES segment(id)
    )
    """
)
conn.execute("""
    INSERT INTO segment (id, type, name, created_at, latest_version_no)
    VALUES (?, ?, ?, ?, ?)""", 
    ("persona1", "persona", "terse-helper", "now", 1))

conn.execute("""
    INSERT INTO segment_version 
             (segment_id, version_no, content,
             role_hint, created_at)
    VALUES (?, ?, ?, ?, ?)""", 
    ("persona1", 1, "You are a terse-helper", "user",
      "2026-06-11T15:31:25.937629"))

seg2 = Segment(id="persona2", type=SegmentType.PERSONA,
                name="angry-helper",
        created_at=datetime.now(), latest_version_no=1)

conn.execute("""
    INSERT INTO segment (id, type, name, created_at, latest_version_no)
    VALUES (?, ?, ?, ?, ?)""", segment_to_row(seg2))

seg3 = row_to_segment(conn.execute("""
    SELECT * FROM segment WHERE id = ?
    """, ("persona2", )).fetchone())
print(seg2 == seg3)


conn.commit()

append_version(conn, "persona1", 2, "talk like a pirate")
try:
    append_version(conn, "persona1", 4, "talk like a gentleman")
except ValueError as e:
    print(e)

conn.commit()

src = SegmentVersion(
    segment_id="persona1", version_no=4, content="refined",
    role_hint="user", derived_from=("persona1", 3),
    created_at=datetime.now(),
)
conn.execute("""
    INSERT INTO segment_version (segment_id, version_no, content,
    role_hint, token_estimate, derived_from, created_at)
    VALUES (?,?,?,?,?,?,?)""", version_to_row(src))
back = row_to_version(conn.execute(
    "SELECT * FROM segment_version WHERE segment_id=? AND version_no=?",
    ("persona1", 4)).fetchone())
assert back == src, (back, src)
print("derived_from round-trip:",
    back.derived_from,
    type(back.derived_from))


conn.commit()

try:
    conn.execute("""
        INSERT INTO segment_version (segment_id, version_no, content) VALUES 
        (?,?,?)""", ("ghost", 1, "orphan"))
    print("FK NOT enforced — orphan inserted")
except sqlite3.IntegrityError as e:
    print("FK enforced:", e)

try:
    conn.execute("""
    INSERT INTO segment_version 
             (segment_id, version_no, content)
    VALUES (?, ?, ?)""", 
    ("persona2", 1, "You are a angry teacher"))
except Exception as e:
     print(e)

try:
    conn.execute("""
        INSERT INTO segment (id, type, name, created_at, latest_version_no)
        VALUES (?, ?, ?, ?, ?)""", 
        ("persona1", "persona", "terse-helper2", "now", 1))
except Exception as e:
     print(str(e))

cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
print([el[0] for el in cur.fetchall()])

for row in conn.execute("SELECT * FROM segment"):
    print([f"{key}={val}" for (val,key) in zip(row,row.keys())])

for row in conn.execute("SELECT * FROM segment_version"):
    print([f"{key}={val}" for (val,key) in zip(row,row.keys())])



conn.close()