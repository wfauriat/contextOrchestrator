from dataclasses import dataclass 
from typing import Any, Protocol
from enum import Enum

@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

@dataclass(frozen=True)
class UserMessage: 
    content: str
@dataclass(frozen=True)
class AssistantMessage: 
    content: str 
    tool_calls: tuple[ToolCall, ...] = ()
@dataclass(frozen=True)
class ToolResultMessage: 
    tool_call: ToolCall
    content: str

Message = UserMessage | AssistantMessage | ToolResultMessage

class StopReason(Enum):
    END = "end"
    TOOL = "tool"
    MAX_TOKENS = "max_tokens"

@dataclass(frozen=True, kw_only=True)
class ChatResult:
    stop_reason: StopReason
    content: str
    tokens_in: int
    tokens_out: int
    tool_calls: tuple[ToolCall, ...] = ()

class Backend(Protocol):
    def call_model(self,
        messages: list[Message],
        *,
        system: str | None = None,
        ) -> ChatResult: ...
    
class BackendError(Exception): pass
class BackendConnectionError(BackendError): pass
class BackendResponseError(BackendError): pass
class BackendContractError(BackendError): pass
