"""Scratch harness to ground the assembler in practice.

Run:  /opt/venvs/pyDS/bin/python scratch_assembly.py
The assembler body is yours to fill in (orchestrator/assembly.py); until then
`window` is None. Everything around it — the store, resolver, spec — is wired.
"""
from datetime import datetime

from orchestrator.domain import (
    Segment, SegmentVersion, SegmentRef, SegmentType,
    ResolvedSegment, AssemblySpec, GenerationParams,
)
from orchestrator.assembly import ConcatV1Assembler

# --- in-memory store: the spike's stand-in for the future Repository ---------
# Two dicts because the domain splits identity from content:
#   segments: id            -> Segment        (carries the TYPE used for routing)
#   versions: (id, version) -> SegmentVersion (carries the CONTENT + role_hint)
segments: dict[str, Segment] = {}
versions: dict[tuple[str, int], SegmentVersion] = {}

def add(seg: Segment, ver: SegmentVersion) -> None:
    segments[seg.id] = seg
    versions[(ver.segment_id, ver.version_no)] = ver

# author a persona (-> system channel) and a task spec (-> a user message)
add(
    Segment(id="persona1", type=SegmentType.PERSONA, name="terse-helper",
            created_at=datetime.now(), latest_version_no=1),
    SegmentVersion(segment_id="persona1", version_no=1,
                   content="You are a terse, helpful assistant.",
                   created_at=datetime.now()),
)
add(
    Segment(id="persona2", type=SegmentType.PERSONA, name="pirate",
            created_at=datetime.now(), latest_version_no=1),
    SegmentVersion(segment_id="persona2", version_no=1,
                   content="Talk like a pirate.",
                   created_at=datetime.now()),
)
add(
    Segment(id="task", type=SegmentType.TASK_SPEC, name="capital-q",
            created_at=datetime.now(), latest_version_no=1),
    SegmentVersion(segment_id="task", version_no=1,
                   content="What is the capital of France?",
                   created_at=datetime.now()),
)
add(
    Segment(id="skill1", type=SegmentType.SKILL, name="mds",
            created_at=datetime.now(), latest_version_no=1),
    SegmentVersion(segment_id="skill1", version_no=1,
                   content="You always output markdowns",
                   created_at=datetime.now()),
)

# --- the resolver callable: SegmentRef -> ResolvedSegment --------------------
# This IS the seam. Today it reads the dicts; at M2 a Repository implements the
# same Callable[[SegmentRef], ResolvedSegment] and the assembler never changes.
# Note the join: type from the Segment, content+role from the SegmentVersion.
def resolve(ref: SegmentRef) -> ResolvedSegment:
    seg = segments[ref.segment_id]
    ver = versions[(ref.segment_id, ref.version_no)]
    return ResolvedSegment(type=seg.type, content=ver.content,
                           role_hint=ver.role_hint, ref=ref)

# --- the recipe (references, not text) --------------------------------------
spec = AssemblySpec(
    id="demo",
    ordered_refs=[
        SegmentRef(segment_id="persona1", version_no=1),
        SegmentRef(segment_id="persona2", version_no=1),
        SegmentRef(segment_id="task", version_no=1),
        SegmentRef(segment_id="skill1", version_no=1),

    ],
    params=GenerationParams(max_tokens=256, temperature=0.0),
    policy_id="concat-v1",
    created_at=datetime.now(),
)

# --- assemble ---------------------------------------------------------------
assembler = ConcatV1Assembler(resolve)
window = assembler.assemble(spec)

print("=== assembled window ===")
print("system  :", None if window is None else window.system)
print("messages:", None if window is None else window.messages)


# --- fire the real model call (the seam) ------------------------------------
from agentAPI.ollama_backend import OllamaBackend
# from agentAPI.anthropic_backend import AnthropicBackend
# from agentAPI.mistral_backend import MistralBackend   # neutral seam — swap freely

backend = OllamaBackend()          # reads ANTHROPIC_API_KEY via load_dotenv()
result = backend.call_model(window.messages, system=window.system)

print("\n=== completion ===")
print("stop_reason  :", result.stop_reason)
print("tokens in/out:", result.tokens_in, "/", result.tokens_out)
print("content      :", result.content)