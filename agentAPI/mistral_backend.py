import os
import json
from typing import Any, Protocol, Iterable, assert_never
import logging
import httpx

from dotenv import load_dotenv

from .backend import (
    UserMessage, AssistantMessage, ToolResultMessage, Message,
    ToolCall, StopReason, ChatResult,
    BackendConnectionError, BackendResponseError
    )

from .tools import REGISTRY, Tool

logger = logging.getLogger(__name__)


class _Response(Protocol):
    def raise_for_status(self) -> Any: ...
    def json(self) -> Any: ...

class _HttpClient(Protocol):
    def post(self,
    url: str, *, json: Any, timeout: float) -> _Response: ...


DEFAULT_SYSTEM = (
    "you are a helpful assistant that answers questions and"
    " provides information. You have access to tools,"
    "use them if you feel it is appropriate or you are asked to."
)


_STOP_REASONS = {
    "stop": StopReason.END,
    "tool_calls": StopReason.TOOL,
    "length": StopReason.MAX_TOKENS,
    "model_length": StopReason.MAX_TOKENS,
}


def _to_mistral_messages(messages: list[Message]) -> list[dict[str, Any]]:
    msg_list = []
    for m in messages:
        match m:
            case UserMessage(content=c):
                msg_list.append({"role": "user", "content": c})
            case AssistantMessage(content=c, tool_calls=tc):
                if tc:
                    tool_dict = [{"id": call.id,
                                  "type": "function",
                                  "function": {
                                      "name": call.name,
                                      "arguments": json.dumps(call.arguments)}}
                                 for call in tc]
                    msg_list.append({"role": "assistant", "content": c,
                                     "tool_calls": tool_dict})
                else:
                    msg_list.append({"role": "assistant", "content": c})
            case ToolResultMessage(tool_call=tc, content=c):
                msg_list.append({"role": "tool", "content": c,
                                 "tool_call_id": tc.id,
                                 "name": tc.name})
            case _:
                assert_never(m)

    return msg_list


def _to_mistral_tools(tools: Iterable[Tool]) -> list[dict[str, Any]]:
    return [{"type": "function",
              "function": {"name": t.name,
                           "description": t.description,
                            "parameters": t.parameters}}
        for t in tools]

class MistralBackend():
    def __init__(self, *,
                 system_prompt: str | None = None,
                 client: _HttpClient | None = None):
        self._url = "https://api.mistral.ai/v1/chat/completions"
        self._model = "mistral-small-latest"
        self._max_tokens = 1024
        self._request_timeout = 60
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        if client is None:
            load_dotenv()
            client = httpx.Client(
                headers={"Authorization":
                         f"Bearer {os.environ['MISTRAL_API_KEY']}"})
        self.client = client

    def call_model(self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> ChatResult:
        """
        Send messages to the Mistral chat-completions endpoint and return
        the assistant reply.

        Returns:
            ChatResult: frozen dataclass (stop_reason, content, tokens, tools).

        Raises:
            BackendConnectionError: transport failed (couldn't reach Mistral).
            BackendResponseError: Mistral returned a non-2xx status.
        """
        if system is None: system = self.system_prompt
        payload = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "stream": False,
            "messages": [{"role": "system",
                          "content": system},
                         *_to_mistral_messages(messages)],
            "tools": _to_mistral_tools(REGISTRY.values()),
            "tool_choice": "auto",
        }
        tools_list = ()
        try:
            logger.debug("Attempting connection with mistral")
            response = self.client.post(self._url, json=payload,
                                        timeout=self._request_timeout)
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            usage = data["usage"]
            logger.debug(
                "Response received from mistral, "
                "Tokens in: %d, Tokens out: %d",
                usage["prompt_tokens"], usage["completion_tokens"])
            if message.get("tool_calls"):
                tools_list = tuple([ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"]["arguments"])
                    ) for tc in message["tool_calls"]
                ])
                logger.debug(tools_list)
            stop_reason = _STOP_REASONS[choice["finish_reason"]]

            output = ChatResult(
                stop_reason=stop_reason,
                content=message["content"] or "",
                tokens_in=usage["prompt_tokens"],
                tokens_out=usage["completion_tokens"],
                tool_calls=tools_list)
            return output

        except httpx.HTTPStatusError as e:
            raise BackendResponseError(
                f"Mistral returned HTTP {e.response.status_code}: "
                f"{e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise BackendConnectionError(
                f"Could not reach Mistral at {self._url}") from e
