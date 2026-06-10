from typing import Any

import pytest
import httpx

from agentAPI.ollama_backend import (OllamaBackend, _to_ollama_messages,
    _to_ollama_tools)
from agentAPI.tools import Tool
from agentAPI.backend import (Message, StopReason, ToolCall,
    UserMessage, AssistantMessage, ToolResultMessage,
    BackendConnectionError, BackendResponseError)

MOCK_MESSAGE: list[Message] = [UserMessage("hello")]

# Mirrors a real Ollama /api/chat response (note: done_reason is always
# present on a completed non-streaming response).
HAPPY_DATA = {
    "message": {"content": "hi"},
    "done_reason": "stop",
    "prompt_eval_count": 12,
    "eval_count": 4,
}

# A tool-call response: content empty, the payload lives in tool_calls,
# done_reason is still "stop" (so detection is by presence, not reason).
TOOL_DATA = {
    "message": {
        "content": "",
        "tool_calls": [
            {"id": "call_abc",
             "function": {"name": "get_weather",
                          "arguments": {"city": "Paris"}}},
        ],
    },
    "done_reason": "stop",
    "prompt_eval_count": 182,
    "eval_count": 20,
}

class FakeResponse():
    def __init__(self, *, status_error=None, data=None):
        self._status_error = status_error
        self._data = HAPPY_DATA if data is None else data
    def raise_for_status(self):
        if self._status_error:
            request = httpx.Request("POST", "http://test")
            response = httpx.Response(self._status_error, text="server boom",
                                      request=request)
            http_err = httpx.HTTPStatusError(f"Error {self._status_error}",
                                             request=request,
                                             response=response)
            raise http_err
    def json(self):
        return self._data

class FakeClient():
    def __init__(self, *, post_error=None, response=None):
        self._post_error = post_error
        self._response = response or FakeResponse()
    def post(self, url: str, *, json: dict[str, Any],
             timeout: float) -> FakeResponse:
        del url, timeout  # required by the _HttpClient protocol; unused here
        self.sent_payload = json
        if self._post_error: raise self._post_error
        return self._response


def test_call_model_parses_successful_response():
    ollama_instance = OllamaBackend(client=FakeClient())
    reply = ollama_instance.call_model(messages=MOCK_MESSAGE)
    assert reply.content == "hi", f"Expected 'hi', got {reply.content!r}"
    assert reply.tokens_in == 12, f"Expected 12, got {reply.tokens_in!r}"
    assert reply.tokens_out == 4, f"Expected 4, got {reply.tokens_out!r}"
    assert reply.stop_reason == StopReason.END, \
        f"Expected END, got {reply.stop_reason!r}"
    assert reply.tool_calls == (), f"Expected (), got {reply.tool_calls!r}"

def test_call_model_parses_tool_call_response():
    client = FakeClient(response=FakeResponse(data=TOOL_DATA))
    reply = OllamaBackend(client=client).call_model(messages=MOCK_MESSAGE)
    assert reply.stop_reason == StopReason.TOOL, \
        f"Expected TOOL, got {reply.stop_reason!r}"
    assert reply.content == "", f"Expected '', got {reply.content!r}"
    assert reply.tool_calls == (
        ToolCall(id="call_abc", name="get_weather",
                 arguments={"city": "Paris"}),
    ), f"got {reply.tool_calls!r}"

def test_call_model_sets_max_tokens_stop_reason_on_length():
    data = {**HAPPY_DATA, "done_reason": "length"}
    client = FakeClient(response=FakeResponse(data=data))
    reply = OllamaBackend(client=client).call_model(messages=MOCK_MESSAGE)
    assert reply.stop_reason == StopReason.MAX_TOKENS, \
        f"Expected MAX_TOKENS, got {reply.stop_reason!r}"

def test_chat_raises_connection_error_when_post_fails():
    ollama_instance = OllamaBackend(
        client=FakeClient(post_error=httpx.RequestError("boom")))
    with pytest.raises(BackendConnectionError) as excinfo:
        ollama_instance.call_model(messages=MOCK_MESSAGE)
    assert "Could not reach" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.RequestError)

def test_chat_raises_status_error_when_response_send_wrong_status():
    ollama_instance = OllamaBackend(
             client=FakeClient(response=FakeResponse(status_error=500)))
    with pytest.raises(BackendResponseError) as excinfo:
        ollama_instance.call_model(messages=MOCK_MESSAGE)
    assert "500" in str(excinfo.value)
    assert "server boom" in str(excinfo.value)

def test_call_model_sends_resolved_system_message_in_payload():
    client = FakeClient()
    backend = OllamaBackend(client=client, system_prompt="default sys")
    backend.call_model(messages=MOCK_MESSAGE, system="be terse")
    sent = client.sent_payload
    assert sent["messages"][0] == {"role": "system", "content": "be terse"}
    assert sent["messages"][1:] == [{"role": "user", "content": "hello"}]


def test_to_ollama_messages_renders_tool_round_trip():
    tc = ToolCall(id="call_1", name="get_weather", arguments={"city": "Paris"})
    history = [
        UserMessage("weather?"),
        AssistantMessage("let me check", (tc,)),
        ToolResultMessage(tc, "sunny"),
    ]
    assert _to_ollama_messages(history) == [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "let me check",
         "tool_calls": [{"function": {"name": "get_weather",
                                      "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "sunny", "tool_name": "get_weather"},
    ]


def test_to_ollama_tools_renders_openai_envelope():
    tool = Tool(name="t", description="d",
                parameters={"type": "object",
                            "properties": {"x": {"type": "string"}},
                            "required": ["x"]},
                func=lambda: "")
    # parameters must nest INSIDE function (the bug was it landing as a sibling)
    assert _to_ollama_tools([tool]) == [{
        "type": "function",
        "function": {"name": "t", "description": "d",
                     "parameters": {"type": "object",
                                    "properties": {"x": {"type": "string"}},
                                    "required": ["x"]}},
    }]
