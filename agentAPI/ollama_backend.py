from typing import Any, Protocol, Iterable, assert_never
import logging
import httpx

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


def _to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    msg_list = []
    for m in messages:
        match m:
            case UserMessage(content=c):
                msg_list.append({"role": "user", "content": c})
            case AssistantMessage(content=c, tool_calls= tc):
                if tc:
                    tool_dict = [{"function": {"name": call.name,
                                            "arguments": call.arguments}}
                                for call in tc]
                    msg_list.append({"role": "assistant", "content": c,
                                     "tool_calls": tool_dict})
                else:
                    msg_list.append({"role": "assistant", "content": c})                   
            case ToolResultMessage(tool_call=tc, content=c):
                msg_list.append({"role": "tool", "content":c,
                                 "tool_name": tc.name})
            case _:
                assert_never(m)

    return msg_list

def _to_ollama_tools(tools: Iterable[Tool]) -> list[dict[str, Any]]:
    return [{"type": "function",
              "function": {"name": t.name,
                           "description": t.description,
                            "parameters": t.parameters}}
        for t in tools]


class OllamaBackend():
    def __init__(self, *,
                 system_prompt: str | None = None,
                 client: _HttpClient | None = None):
        self._url = "http://localhost:11434/api/chat"
        self._model = "qwen3:8b"
        self._think = False
        self._request_timeout = 60
        self.system_prompt = system_prompt or DEFAULT_SYSTEM
        self.client = client or httpx.Client()

    def call_model(self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> ChatResult:
        """
        Send messages to the local Ollama model and return the 
        assistant reply

        Returns:
            ChatResult: frozen dataclass (content, tokens_in, tokens_out).

        Raises:
            BackendConnectionError: transport failed (couldn't reach Ollama).
            BackendResponseError: Ollama returned a non-2xx status.
        """
        if system is None: system = self.system_prompt
        payload = {
            "model": self._model,
            "stream": False,
            "think": self._think,
            "messages": [{"role": "system",
                          "content": system},
                          *_to_ollama_messages(messages)],
            "tools": _to_ollama_tools(REGISTRY.values()),
        }
        tools_list = ()
        try:
            logger.debug("Attempting connection with ollama")
            response = self.client.post(self._url, json=payload,
                                    timeout=self._request_timeout)
            response.raise_for_status()
            data = response.json()
            logger.debug(
                "Response received from ollama, "
                "Tokens in: %d, Tokens out: %d",
                data["prompt_eval_count"], data["eval_count"])
            if data["message"].get("tool_calls"):
                stop_reason = StopReason.TOOL
                tools_list = tuple([ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"]
                    ) for tc in data["message"]["tool_calls"]
                ])
                logger.debug(tools_list)
            elif data["done_reason"] == "length":
                stop_reason = StopReason.MAX_TOKENS
            else: stop_reason = StopReason.END

            output = ChatResult(
                stop_reason=stop_reason,
                content=data["message"]["content"],
                tokens_in=data["prompt_eval_count"],
                tokens_out=data["eval_count"],
                tool_calls=tools_list)
            return output
        
        except httpx.HTTPStatusError as e:
            raise BackendResponseError(
                f"Ollama returned HTTP {e.response.status_code}: "
                f"{e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise BackendConnectionError(
                f"Could not reach Ollama at {self._url}") from e
        

