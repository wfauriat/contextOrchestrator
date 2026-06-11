from enum import Enum
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Literal


class SegmentType(str, Enum):
    PERSONA = "persona"
    SKILL = "skill"
    KNOWLEDGE = "knowledge"
    TASK_SPEC = "task_spec"
    WORKING_MEMORY = "working_memory"

class SegmentVersion(BaseModel):
    model_config = ConfigDict(frozen=True)
    segment_id: str
    version_no: int
    content: str
    role_hint: Literal["user", "assistant"] = "user"
    token_estimate: int | None = None
    derived_from: tuple[str, int] | None = None
    created_at: datetime

class Segment(BaseModel):
    id: str
    type: SegmentType
    name: str
    created_at: datetime
    latest_version_no: int | None

class SegmentRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    segment_id: str
    version_no: int

class ResolvedSegment(BaseModel):
    type: SegmentType
    content: str
    role_hint: Literal["user", "assistant"]
    ref: SegmentRef

class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()

class AssemblySpec(BaseModel):
    id: str
    ordered_refs: list[SegmentRef]
    params: GenerationParams
    policy_id: str
    created_at: datetime
