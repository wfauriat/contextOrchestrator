from dataclasses import dataclass
from typing import Callable

from agentAPI.backend import (
    Message, UserMessage, AssistantMessage
)
from orchestrator.domain import (
    AssemblySpec, SegmentRef, SegmentType, ResolvedSegment
)

@dataclass
class AssembledWindow:
    system: str | None
    messages: list[Message]

Resolver = Callable[[SegmentRef], ResolvedSegment]

class ConcatV1Assembler:
    def __init__(self, resolve: Resolver):
        self._resolve = resolve
    
    def assemble(self, spec: AssemblySpec) -> AssembledWindow:
        persona_parts = []
        messages: list[Message] = []
        SEP = "\n"
        for ref in spec.ordered_refs:
            r = self._resolve(ref)
            if r.type == SegmentType.PERSONA:
                persona_parts.append(r.content)
            else:
                if r.role_hint == "user":
                    messages.append(UserMessage(r.content))
                elif r.role_hint == "assistant":
                    messages.append(AssistantMessage(r.content))
        system = SEP.join(persona_parts) if persona_parts else None
        return AssembledWindow(system=system, messages=messages)