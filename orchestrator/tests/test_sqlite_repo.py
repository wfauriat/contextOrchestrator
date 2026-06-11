"""Tests for the SQLite Repository adapter.

Substrate idiom: module-level test_* functions, small hand-rolled builders,
no test classes. Each test gets a fresh in-memory DB via _repo().
"""
from datetime import datetime

import pytest

from orchestrator.domain import Segment, SegmentVersion, SegmentType
from orchestrator.ports import SegmentNotFound
from orchestrator.sqlite_repo import SqliteRepository

_T = datetime(2026, 1, 1, 12, 0, 0)


def _repo() -> SqliteRepository:
    return SqliteRepository.connect()  # :memory:


def _segment(id="p1", type=SegmentType.PERSONA, name="terse") -> Segment:
    return Segment(id=id, type=type, name=name, created_at=_T, latest_version_no=None)


def _version(segment_id="p1", version_no=1, content="hi", **kw) -> SegmentVersion:
    return SegmentVersion(
        segment_id=segment_id, version_no=version_no, content=content,
        created_at=_T, **kw,
    )


# --- round-trip: the encode/decode are inverses ------------------------------

def test_create_and_get_segment_round_trips():
    repo = _repo()
    created = repo.create_segment(_segment())
    assert created == repo.get_segment("p1")
    assert created.type is SegmentType.PERSONA


def test_version_round_trips_including_derived_from_tuple():
    repo = _repo()
    repo.create_segment(_segment())
    repo.append_version(_version(version_no=1))
    src = _version(version_no=2, content="refined", derived_from=("p1", 1),
                   token_estimate=7, role_hint="assistant")
    stored = repo.append_version(src)
    back = repo.get_version("p1", 2)
    assert back == src == stored
    assert isinstance(back.derived_from, tuple)   # not a list (json gives lists)


# --- the append invariants ---------------------------------------------------

def test_append_bumps_latest_version_no():
    repo = _repo()
    repo.create_segment(_segment())
    repo.append_version(_version(version_no=1))
    repo.append_version(_version(version_no=2))
    assert repo.get_segment("p1").latest_version_no == 2


def test_first_version_must_be_one():
    repo = _repo()
    repo.create_segment(_segment())
    with pytest.raises(ValueError):
        repo.append_version(_version(version_no=2))


def test_non_monotonic_version_rejected():
    repo = _repo()
    repo.create_segment(_segment())
    repo.append_version(_version(version_no=1))
    with pytest.raises(ValueError):
        repo.append_version(_version(version_no=3))


def test_rejected_append_leaves_no_partial_state():
    """Atomicity: the failed insert and the latest_version_no bump both roll back."""
    repo = _repo()
    repo.create_segment(_segment())
    repo.append_version(_version(version_no=1))
    with pytest.raises(ValueError):
        repo.append_version(_version(version_no=5))
    # version 5 was never written, and latest stayed at 1
    assert repo.get_segment("p1").latest_version_no == 1
    with pytest.raises(SegmentNotFound):
        repo.get_version("p1", 5)


# --- not-found and referential integrity -------------------------------------

def test_get_missing_segment_raises():
    with pytest.raises(SegmentNotFound):
        _repo().get_segment("ghost")


def test_get_missing_version_raises():
    repo = _repo()
    repo.create_segment(_segment())
    with pytest.raises(SegmentNotFound):
        repo.get_version("p1", 1)


def test_append_to_unknown_segment_raises():
    with pytest.raises(SegmentNotFound):
        _repo().append_version(_version(segment_id="ghost", version_no=1))


def test_duplicate_segment_id_rejected():
    import sqlite3
    repo = _repo()
    repo.create_segment(_segment())
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_segment(_segment())


# --- listing -----------------------------------------------------------------

def test_list_segments_filters_by_type():
    repo = _repo()
    repo.create_segment(_segment(id="p1", type=SegmentType.PERSONA))
    repo.create_segment(_segment(id="k1", type=SegmentType.KNOWLEDGE, name="facts"))
    assert {s.id for s in repo.list_segments()} == {"p1", "k1"}
    assert [s.id for s in repo.list_segments(SegmentType.KNOWLEDGE)] == ["k1"]