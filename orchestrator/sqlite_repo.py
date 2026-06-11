"""SQLite adapter for the Repository port.

The mapping functions (`*_to_row` / `row_to_*`) are the ones validated in the
M1 scratch session, promoted verbatim. They stay module-level and pure so they
can be tested without a database. The class owns connection lifecycle, schema
bootstrap, and the transactional invariants.
"""
import json
import sqlite3
from datetime import datetime

from .domain import Segment, SegmentVersion, SegmentType
from .ports import Repository, SegmentNotFound


# --- schema -----------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS segment (
    id                  TEXT PRIMARY KEY,
    type                TEXT NOT NULL,
    name                TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    latest_version_no   INTEGER
);

CREATE TABLE IF NOT EXISTS segment_version (
    segment_id          TEXT NOT NULL,
    version_no          INTEGER NOT NULL,
    content             TEXT NOT NULL,
    role_hint           TEXT NOT NULL,
    token_estimate      INTEGER,
    derived_from        TEXT,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (segment_id, version_no),
    FOREIGN KEY (segment_id) REFERENCES segment(id)
);
"""


# --- row <-> model mapping (pure; validated in scratch) ---------------------

def segment_to_row(seg: Segment) -> tuple:
    return (
        seg.id,
        seg.type.value,
        seg.name,
        seg.created_at.isoformat(),
        seg.latest_version_no,
    )


def row_to_segment(row: sqlite3.Row) -> Segment:
    return Segment(
        id=row["id"],
        type=row["type"],                       # pydantic coerces str -> SegmentType
        name=row["name"],
        created_at=datetime.fromisoformat(row["created_at"]),
        latest_version_no=row["latest_version_no"],
    )


def version_to_row(v: SegmentVersion) -> tuple:
    return (
        v.segment_id,
        v.version_no,
        v.content,
        v.role_hint,
        v.token_estimate,
        json.dumps(list(v.derived_from)) if v.derived_from else None,
        v.created_at.isoformat(),
    )


def row_to_version(row: sqlite3.Row) -> SegmentVersion:
    raw = row["derived_from"]
    return SegmentVersion(
        segment_id=row["segment_id"],
        version_no=row["version_no"],
        content=row["content"],
        role_hint=row["role_hint"],
        token_estimate=row["token_estimate"],
        derived_from=tuple(json.loads(raw)) if raw else None,   # rebuild tuple, not list
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# --- adapter ----------------------------------------------------------------

class SqliteRepository(Repository):
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")  # per-connection; before any txn
        self._conn.executescript(_SCHEMA)               # idempotent (IF NOT EXISTS)
        self._conn.commit()

    @classmethod
    def connect(cls, path: str = ":memory:") -> "SqliteRepository":
        """Convenience constructor. Default is an in-memory DB (tests)."""
        return cls(sqlite3.connect(path))

    # --- segments & versions ---

    def create_segment(self, segment: Segment) -> Segment:
        with self._conn:
            self._conn.execute(
                """INSERT INTO segment (id, type, name, created_at, latest_version_no)
                   VALUES (?, ?, ?, ?, ?)""",
                segment_to_row(segment),
            )
        return self.get_segment(segment.id)

    def get_segment(self, segment_id: str) -> Segment:
        row = self._conn.execute(
            "SELECT * FROM segment WHERE id = ?", (segment_id,)
        ).fetchone()
        if row is None:
            raise SegmentNotFound(f"segment {segment_id!r}")
        return row_to_segment(row)

    def append_version(self, version: SegmentVersion) -> SegmentVersion:
        with self._conn:
            row = self._conn.execute(
                "SELECT latest_version_no FROM segment WHERE id = ?",
                (version.segment_id,),
            ).fetchone()
            if row is None:
                raise SegmentNotFound(f"segment {version.segment_id!r}")

            expected = (row["latest_version_no"] or 0) + 1   # None -> first version is 1
            if version.version_no != expected:
                raise ValueError(
                    f"non-monotonic version: got {version.version_no}, expected {expected}"
                )

            self._conn.execute(
                """INSERT INTO segment_version
                   (segment_id, version_no, content, role_hint,
                    token_estimate, derived_from, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                version_to_row(version),
            )
            self._conn.execute(
                "UPDATE segment SET latest_version_no = ? WHERE id = ?",
                (version.version_no, version.segment_id),
            )
        return self.get_version(version.segment_id, version.version_no)

    def get_version(self, segment_id: str, version_no: int) -> SegmentVersion:
        row = self._conn.execute(
            "SELECT * FROM segment_version WHERE segment_id = ? AND version_no = ?",
            (segment_id, version_no),
        ).fetchone()
        if row is None:
            raise SegmentNotFound(f"version {version_no} of segment {segment_id!r}")
        return row_to_version(row)

    def list_segments(self, type: SegmentType | None = None) -> list[Segment]:
        if type is None:
            rows = self._conn.execute(
                "SELECT * FROM segment ORDER BY created_at"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM segment WHERE type = ? ORDER BY created_at",
                (type.value,),
            ).fetchall()
        return [row_to_segment(r) for r in rows]