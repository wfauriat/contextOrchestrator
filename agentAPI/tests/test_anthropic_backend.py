from collections import namedtuple

import pytest
import httpx
import anthropic
from anthropic.types import TextBlock, ToolUseBlock

from agentAPI.anthropic_backend import (AnthropicBackend,
    _to_anthropic_messages, _to_anthropic_tools)
from agentAPI.tools import Tool
from agentAPI.backend import (Message, StopReason, ToolCall,
    UserMessage, AssistantMessage, ToolResultMessage,
    BackendConnectionError, BackendResponseError)

MOCK_MESSAGE: list[Message] = [UserMessage("hello")]


class FakeMessage():
    def __init__(self, *, create_error=None, content=None,
                 stop_reason="end_turn"):
        self._create_error = create_error
        self._content = ([TextBlock.model_validate({"type": "text",
                                                     "text": "hi"})]
                         if content is None else content)
        self._stop_reason = stop_reason

    def create(self, **kwargs):
        self.sent_kwargs = kwargs
        if self._create_error:
            raise self._create_error
        Usage = namedtuple("Usage", ["input_tokens", "output_tokens"])
        Response = namedtuple("Response", ["content", "usage", "stop_reason"])
        return Response(content=self._content,
                        usage=Usage(input_tokens=12, output_tokens=4),
                        stop_reason=self._stop_reason)


class FakeClient():
    def __init__(self, message):
        self.messages = message


def _api_status_error(status_code):
    request = httpx.Request("POST", "http://test")
    response = httpx.Response(status_code, text="server boom",
                             request=request)
    return anthropic.APIStatusError("server boom",
                                    response=response, body=None)


def test_call_model_parses_successful_response():
    backend = AnthropicBackend(client=FakeClient(FakeMessage()))
    reply = backend.call_model(messages=MOCK_MESSAGE)
    assert reply.stop_reason == StopReason.END
    assert reply.content == "hi", f"Expected 'hi', got {reply.content!r}"
    assert reply.tokens_in == 12, f"Expected 12, got {reply.tokens_in!r}"
    assert reply.tokens_out == 4, f"Expected 4, got {reply.tokens_out!r}"
    assert reply.tool_calls == ()


def test_call_model_parses_tool_use_response():
    tool_block = ToolUseBlock(id="toolu_1", name="get_weather",
                              input={"city": "Paris"}, type="tool_use")
    backend = AnthropicBackend(client=FakeClient(
        FakeMessage(content=[tool_block], stop_reason="tool_use")))
    reply = backend.call_model(messages=MOCK_MESSAGE)
    assert reply.stop_reason == StopReason.TOOL
    assert reply.content == ""
    assert reply.tool_calls == (
        ToolCall(id="toolu_1", name="get_weather",
                 arguments={"city": "Paris"}),)


def test_call_model_maps_max_tokens_stop_reason():
    backend = AnthropicBackend(client=FakeClient(
        FakeMessage(stop_reason="max_tokens")))
    reply = backend.call_model(messages=MOCK_MESSAGE)
    assert reply.stop_reason == StopReason.MAX_TOKENS


def test_call_model_raises_connection_error_when_create_fails():
    request = httpx.Request("POST", "http://test")
    backend = AnthropicBackend(client=FakeClient(
        FakeMessage(create_error=anthropic.APIConnectionError(
            message="boom", request=request))))
    with pytest.raises(BackendConnectionError) as excinfo:
        backend.call_model(messages=MOCK_MESSAGE)
    assert "Could not reach" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, anthropic.APIConnectionError)


def test_call_model_raises_response_error_on_api_status_error():
    backend = AnthropicBackend(client=FakeClient(
        FakeMessage(create_error=_api_status_error(429))))
    with pytest.raises(BackendResponseError) as excinfo:
        backend.call_model(messages=MOCK_MESSAGE)
    assert "429" in str(excinfo.value)


def test_call_model_sends_system_and_messages_in_payload():
    # no per-call system -> falls back to the constructor default
    message = FakeMessage()
    backend = AnthropicBackend(system_prompt="be terse",
                               client=FakeClient(message))
    backend.call_model(messages=MOCK_MESSAGE)
    sent = message.sent_kwargs
    assert sent["system"] == "be terse"
    assert sent["messages"] == [{"role": "user", "content": "hello"}]


def test_call_model_per_call_system_overrides_default():
    # default and override differ, so the assertion proves the override won
    message = FakeMessage()
    backend = AnthropicBackend(system_prompt="default sys",
                               client=FakeClient(message))
    backend.call_model(messages=MOCK_MESSAGE, system="be terse")
    sent = message.sent_kwargs
    assert sent["system"] == "be terse"
    assert sent["messages"] == [{"role": "user", "content": "hello"}]


def test_to_anthropic_messages_renders_tool_round_trip():
    tc = ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Paris"})
    history = [
        UserMessage("weather?"),
        AssistantMessage("let me check", (tc,)),
        ToolResultMessage(tc, "sunny"),
    ]
    assert _to_anthropic_messages(history) == [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
             "input": {"city": "Paris"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": "sunny"},
        ]},
    ]


def test_to_anthropic_tools_renders_flat_input_schema():
    tool = Tool(name="t", description="d",
                parameters={"type": "object",
                            "properties": {"x": {"type": "string"}},
                            "required": ["x"]},
                func=lambda: "")
    # flat shape: no function envelope, schema under input_schema not parameters
    assert _to_anthropic_tools([tool]) == [{
        "name": "t", "description": "d",
        "input_schema": {"type": "object",
                         "properties": {"x": {"type": "string"}},
                         "required": ["x"]},
    }]
