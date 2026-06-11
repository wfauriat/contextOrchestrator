"""Ports — the abstract interfaces the system depends on, not their implementations.

Phase-1 scope: only the segment/version half of `Repository` is defined here,
because that's what M1 has built and verified. `save_assembly` / `get_assembly`
and the run methods (design §4.3) land in M2 alongside the layers that need them
— grown into one at a time, not stubbed out ahead of use.
"""
from abc import ABC, abstractmethod

from .domain import Segment, SegmentVersion, SegmentType


class SegmentNotFound(Exception):
    """Raised by get_* when the requested segment/version does not exist.

    A domain-level error so callers never have to know the adapter is sqlite —
    they catch this, not sqlite3.Error or a bare KeyError.
    """


class Repository(ABC):
    """Persistence behind an interface; sqlite now, Postgres a local swap later.

    Convention: write methods RETURN the persisted object, so an implementation
    is free to fill in fields it owns (e.g. a normalized `latest_version_no`)
    and the caller always sees the canonical stored state, not its own input.

    Append-only where the domain is: versions are never updated or deleted.
    """

    # --- segments & versions (M1) ---

    @abstractmethod
    def create_segment(self, segment: Segment) -> Segment:
        """Insert a new segment identity. Raises on duplicate id."""

    @abstractmethod
    def get_segment(self, segment_id: str) -> Segment:
        """Fetch one segment by id. Raises SegmentNotFound if absent."""

    @abstractmethod
    def append_version(self, version: SegmentVersion) -> SegmentVersion:
        """Append the next immutable version.

        Contract: reject version_no != latest+1 (monotonicity, ValueError);
        atomically insert the version AND bump the segment's latest_version_no.
        """

    @abstractmethod
    def get_version(self, segment_id: str, version_no: int) -> SegmentVersion:
        """Fetch one immutable version by its composite key.
        Raises SegmentNotFound if absent."""

    @abstractmethod
    def list_segments(self, type: SegmentType | None = None) -> list[Segment]:
        """All segments, optionally filtered by type."""