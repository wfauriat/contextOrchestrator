from agentAPI.agent import Agent
from agentAPI.backend import (
    Message, ChatResult, StopReason, ToolCall,
    UserMessage, AssistantMessage, ToolResultMessage,
)


def _reply(stop: StopReason, content: str = "",
           tool_calls: tuple[ToolCall, ...] = ()) -> ChatResult:
    return ChatResult(stop_reason=stop, content=content,
                      tokens_in=1, tokens_out=1, tool_calls=tool_calls)


class FakeBackend:
    """Scripted backend: hands back the given ChatResults in order, clamping to
    the last once exhausted — so a single TOOL reply loops until max_rounds.
    Records call count and a snapshot of the transcript seen on each call."""
    def __init__(self, results: list[ChatResult]):
        self._results = results
        self.calls = 0
        self.received: list[list[Message]] = []

    def call_model(self, messages: list[Message], *,
                   system: str | None = None) -> ChatResult:
        del system  # required by the Backend protocol; unused by this fake
        self.received.append(list(messages))
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


def _approve_all(_: ToolCall) -> bool:
    return True


# --- Agent.run: the loop's control flow --------------------------------------

def test_run_with_no_tools_calls_backend_once_and_appends_one_reply():
    backend = FakeBackend([_reply(StopReason.END, "hello")])
    agent = Agent(backend)
    messages: list[Message] = [UserMessage("hi")]

    result = agent.run(messages)

    assert backend.calls == 1, f"got {backend.calls}"
    assert result.content == "hello", f"got {result.content!r}"
    assert messages == [UserMessage("hi"), AssistantMessage("hello")], \
        f"got {messages!r}"

def test_run_recalls_once_after_all_tool_results_then_finishes():
    # Two tool calls in ONE round must trigger exactly one re-call (not one per
    # result), and exactly one final assistant turn — the two control-flow bugs
    # caught in review (per-tool re-call; assistant appended every round).
    tc1 = ToolCall(id="c1", name="unknown_a", arguments={})
    tc2 = ToolCall(id="c2", name="unknown_b", arguments={})
    backend = FakeBackend([
        _reply(StopReason.TOOL, "", (tc1, tc2)),
        _reply(StopReason.END, "done"),
    ])
    agent = Agent(backend, approve=_approve_all)
    messages: list[Message] = [UserMessage("go")]

    result = agent.run(messages)

    assert backend.calls == 2, f"expected 2 backend calls, got {backend.calls}"
    assert result.content == "done", f"got {result.content!r}"
    assert messages == [
        UserMessage("go"),
        AssistantMessage("", (tc1, tc2)),
        ToolResultMessage(tc1, "error: unknown tool unknown_a"),
        ToolResultMessage(tc2, "error: unknown tool unknown_b"),
        AssistantMessage("done"),
    ], f"got {messages!r}"

def test_run_stops_at_max_rounds_when_model_never_finishes():
    tc = ToolCall(id="c", name="unknown", arguments={})
    backend = FakeBackend([_reply(StopReason.TOOL, "", (tc,))])  # always TOOL
    agent = Agent(backend, approve=_approve_all, max_rounds=2)

    result = agent.run([UserMessage("loop")])

    # initial call + 2 rounds, then the cap halts it (no infinite loop).
    assert backend.calls == 3, f"got {backend.calls}"
    assert result.stop_reason == StopReason.TOOL, f"got {result.stop_reason!r}"

def test_run_executes_approved_tool_and_feeds_rendered_result_back():
    tc = ToolCall(id="c1", name="run_bash",
                  arguments={"command": "echo fed-back-marker"})
    backend = FakeBackend([
        _reply(StopReason.TOOL, "", (tc,)),
        _reply(StopReason.END, "final"),
    ])
    agent = Agent(backend, approve=_approve_all)
    messages: list[Message] = [UserMessage("run it")]

    result = agent.run(messages)

    assert result.content == "final", f"got {result.content!r}"
    tool_result = messages[2]
    assert isinstance(tool_result, ToolResultMessage), f"got {tool_result!r}"
    assert "exit 0" in tool_result.content, f"got {tool_result.content!r}"
    assert "fed-back-marker" in tool_result.content, \
        f"got {tool_result.content!r}"


# --- Agent._execute: the safety gate + dispatch + error containment ----------

def test_execute_reports_unknown_tool():
    agent = Agent(FakeBackend([_reply(StopReason.END)]))
    out = agent._execute(ToolCall(id="c", name="nope", arguments={}))
    assert out == "error: unknown tool nope", f"got {out!r}"

def test_execute_blocks_dispatch_when_declined():
    agent = Agent(FakeBackend([_reply(StopReason.END)]),
                  approve=lambda _: False)
    out = agent._execute(ToolCall(id="c", name="run_bash",
                                  arguments={"command": "echo SHOULD_NOT_RUN"}))
    assert out == "user declined to use this command", f"got {out!r}"
    # had it run, the echoed marker would surface in the rendered output.
    assert "SHOULD_NOT_RUN" not in out

def test_execute_runs_approved_tool():
    agent = Agent(FakeBackend([_reply(StopReason.END)]), approve=_approve_all)
    out = agent._execute(ToolCall(id="c", name="run_bash",
                                  arguments={"command": "echo ran-ok"}))
    assert "exit 0" in out and "ran-ok" in out, f"got {out!r}"

def test_execute_catches_dispatch_exception_and_returns_error_string():
    # Wrong kwargs make the dispatch callable raise TypeError; the loop must
    # see a tool-result string, never an exception.
    agent = Agent(FakeBackend([_reply(StopReason.END)]), approve=_approve_all)
    out = agent._execute(ToolCall(id="c", name="run_bash",
                                  arguments={"bad_kwarg": 1}))
    assert out.startswith("error:"), f"got {out!r}"
