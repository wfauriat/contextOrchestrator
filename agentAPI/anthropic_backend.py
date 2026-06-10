from typing import Any, Protocol, Iterable, cast, assert_never
from dotenv import load_dotenv
import logging

import anthropic
from anthropic.types import (
    MessageParam, TextBlock, ToolUseBlock, ToolUnionParam)

from .backend import (
    UserMessage, AssistantMessage, ToolResultMessage, Message, 
    ToolCall, StopReason, ChatResult,
    BackendConnectionError, BackendContractError, BackendResponseError
    )
from .tools import Tool, REGISTRY

logger = logging.getLogger(__name__)

class _Messages(Protocol):
    def create(self, **kwargs: Any) -> Any: ...

class _AnthropicClient(Protocol):
    @property
    def messages(self) -> _Messages: ...

DEFAULT_SYSTEM = (
    "you are a helpful assistant that answers questions and"
    " provides information. You have access to tools,"
    "use them if you feel it is appropriate or you are asked to."
)

_STOP_REASONS = {
    "end_turn": StopReason.END,
    "tool_use": StopReason.TOOL,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.END,
}


def _to_anthropic_messages(messages: list[Message]) -> list[MessageParam]:
    msg_list = []
    for m in messages:
        match m:
            case UserMessage(content=c):
                msg_list.append({"role": "user", "content": c})
            case AssistantMessage(content=c, tool_calls= tc):
                content_block = [{"type": "text", "text": c}] if c else []
                if tc:
                    tool_blocks = [{"type": "tool_use",
                                    "id": call.id,
                                    "name": call.name,
                                    "input": call.arguments}
                                    for call in tc]
                    msg_list.append({"role": "assistant",
                                "content": [*content_block, *tool_blocks]})
                else:
                    msg_list.append({"role": "assistant",
                                     "content": content_block})                   
            case ToolResultMessage(tool_call=tc, content=c):
                msg_list.append({"role": "user",
                                 "content":[{
                                     "type":"tool_result",
                                     "tool_use_id": tc.id,
                                     "content":c,
                                 }]
                            })
            case _:
                assert_never(m)

    return msg_list


def _to_anthropic_tools(tools: Iterable[Tool]) -> list[dict[str, Any]]:
    return [{"name": t.name,
             "description": t.description,
             "input_schema": t.parameters}
        for t in tools]


class AnthropicBackend():
    def __init__(self,
            system_prompt: str | None = None,
            *,
            client: _AnthropicClient | None = None) -> None:
        self._model = "claude-haiku-4-5-20251001"
        self._max_tokens = 1024
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        if client is None:
            load_dotenv()
        self.client = client or anthropic.Anthropic()

    def call_model(self,
            messages: list[Message],
            *,
            system: str | None = None) -> ChatResult:
        if system is None:
            system = self.system_prompt
        try:
            logger.debug("Attempting connection with anthropic")
            response = self.client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                tools=cast(list[ToolUnionParam],
                            _to_anthropic_tools(REGISTRY.values())),
                tool_choice={"type": "auto",
                             "disable_parallel_tool_use": True},
                messages=cast(list[MessageParam], 
                              _to_anthropic_messages(messages))
            )
            logger.debug(
                "Response received from anthropic, "
                "Tokens in: %d, Tokens out: %d",
                response.usage.input_tokens, response.usage.output_tokens)
            text_list: list[TextBlock]= []
            tools_list: list[ToolUseBlock] = []
            for block in response.content:
                if isinstance(block, TextBlock):
                    text_list.append(block)
                if isinstance(block, ToolUseBlock):
                    tools_list.append(block)
            raw_stop = response.stop_reason
            if raw_stop is None:
                raise BackendContractError(
                    "Anthropic response has no stop_reason")
            logger.debug(tools_list)
            return ChatResult(
                    stop_reason=_STOP_REASONS[raw_stop],
                    content="".join(b.text for b in text_list),
                    tokens_in=response.usage.input_tokens,
                    tokens_out=response.usage.output_tokens,
                    tool_calls=tuple([ToolCall(
                        id=tc.id,
                        name=tc.name,
                        arguments=tc.input
                        ) for tc in tools_list
                ]))
        except anthropic.APIConnectionError as e:
            raise BackendConnectionError(
                f"Could not reach Anthropic : {str(e)}") from e
        except anthropic.APIStatusError as e:
            raise BackendResponseError(
                f"Anthropic returned HTTP {e.status_code}: {e.body}"
            ) from e
