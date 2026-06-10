# pyarchAgent — Current-State Spec

> A snapshot of what exists in this repo as of 2026-06-10, written so it can serve
> as the **foundation** for a larger project (LLM context management, A/B testing &
> observability of output quality vs. token budget, persona / skills / ICL selection
> and variation). It describes what is *built and working*, the *design decisions* that
> shaped it, and the *seams* a larger system would plug into.

---

## 1. What this is

`pyarchAgent` is a from-scratch, multi-backend LLM **agent wrapper**: conversation
state, a normalized request/response model across three providers, tool calling, and a
tool-execution loop (`StopReason.TOOL` → execute → feed result back → re-call until
`END`). It currently ships one real tool — `run_bash` — gated behind a human approval
callback.

It was deliberately built incrementally as a software-engineering practice exercise
(SOLID, DI, typing, testing, git hygiene). The consequence relevant to a *new* project:
the abstractions are small, explicit, and well-tested, with clean injection seams — it
is a good substrate to build on rather than a framework to fight.

**Stack:** Python 3.12, `httpx` (Ollama/Mistral, raw HTTP), `anthropic` SDK (Anthropic),
`python-dotenv`, `pytest`, `mypy`/`pyright`. 45 tests passing.

---

## 2. Intent (carried from CLAUDE.md)

- Build a real, useful LLM agent, but as a *pretext for deliberate craft*: Python design,
  bash fluency, git hygiene.
- Two (now three) backends behind a **single interface**:
  - Local: Ollama, `qwen3:8b`, 8 GB VRAM budget.
  - Remote: Anthropic SDK.
  - Added: Mistral (HTTP-only, OpenAI-compatible shape).
- Work mode historically was *tutor-first* (the user writes the code); a few components
  (Mistral backend, `probe_bash.py`) were written in one pass by explicit request.
- Security posture is **relaxed but proportionate**: human-in-the-loop approval per tool
  call; no sandbox/allowlist yet (deferred until unattended execution or untrusted input
  is a real trigger).

---

## 3. Repository layout

```
agentAPI/                    # the package (run via `python -m agentAPI`)
  __init__.py                # public surface (re-exports + __all__)
  __main__.py                # CLI entry: argparse -b {ollama,anthropic,mistral}, logging config
  backend.py                 # NEUTRAL core types + Backend Protocol + error hierarchy
  agent.py                   # Agent: the I/O-free tool loop + thin repl() driver
  tools.py                   # run_bash executor, BashResult.render(), Tool + REGISTRY
  ollama_backend.py          # OllamaBackend  (httpx, native /api/chat)
  anthropic_backend.py       # AnthropicBackend (anthropic SDK)
  mistral_backend.py         # MistralBackend (httpx, OpenAI-compatible)
  tests/                     # 45 tests, hand-rolled fakes at every seam
Makefile                     # chat_ollama / chat_anthropic / chat_mistral / test
pyproject.toml               # pytest config (pythonpath, testpaths)
SESSIONS.md                  # append-only design journal (the "why" behind every decision)
notes.md                     # personal cheatsheets (dataclasses, typing, logging, enum, ...)
audit_python_idoms*.md       # idiom audits
sandboxing_and_primitives.md # notes toward future sandboxing
probe_bash.py                # standalone dry-run (deny-all) tool-intent probe
```

> Path note: this repo was `git subtree split` out of a training repo on 2026-06-03.
> Old commits/log entries may reference `projects/agentBuilding/...` — historical only;
> the directory above is the repo root.

---

## 4. The core abstraction: a vendor-neutral domain model

This is the part most relevant to a larger system. Everything vendor-specific is pushed
to the edges; the middle speaks one neutral vocabulary. (`agentAPI/backend.py`)

### 4.1 Messages — discriminated union (sum type)

```python
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
    tool_calls: tuple[ToolCall, ...] = ()      # empty = pure text; text & calls coexist

@dataclass(frozen=True)
class ToolResultMessage:
    tool_call: ToolCall                          # carries the WHOLE ToolCall (id+name can't drift)
    content: str

Message = UserMessage | AssistantMessage | ToolResultMessage
```

Design decision (SESSIONS, 2026-06-02): the **request** side is a discriminated union
(illegal states unrepresentable), because the fields are mutually exclusive by role.
The **response** side (`ChatResult`, below) is the opposite — one struct with
always-present common fields — because there the common-and-always-present argument wins.
This asymmetry is deliberate and documented.

### 4.2 Result — single struct

```python
class StopReason(Enum):           # neutral vocabulary, NOT vendor wire strings
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
```

`tokens_in` / `tokens_out` are already first-class on every result — **the token-budget
observability the larger project needs is already surfaced at the boundary**, per backend,
per call.

### 4.3 The Backend Protocol (structural, not ABC)

```python
class Backend(Protocol):
    def call_model(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> ChatResult: ...
```

Decision (2026-05-30): `typing.Protocol`, not ABC, because there are **two seams** — the
backend seam (ours↔ours) *and* the client seam (ours↔vendor SDK, faked in tests with
duck types). Only Protocol covers both. The contract is the **intersection** of backend
capabilities, never the union; growth is meant to come via a parameter object
(`GenerationParams` frozen dataclass), not an ever-widening keyword list.

### 4.4 Error hierarchy (transport / protocol / contract taxonomy)

```python
class BackendError(Exception): pass
class BackendConnectionError(BackendError): pass   # transport: couldn't reach server
class BackendResponseError(BackendError): pass     # protocol: non-2xx status
class BackendContractError(BackendError): pass     # contract: 200 but value shape wrong
```

Exception **types** are backend-agnostic; exception **messages** are backend-specific
diagnostics. Every backend translates its vendor errors into these.

---

## 5. Backends — adapters / anti-corruption layer

All three implement `Backend`. Each owns two inbound translators
(`_to_<vendor>_messages`, `_to_<vendor>_tools`: neutral → vendor wire form) and inline
outbound parsing (vendor response → `ChatResult`). The vendor divergence the neutral
layer exists to absorb is concrete:

| | Ollama | Anthropic | Mistral |
|---|---|---|---|
| Transport | `httpx`, native `/api/chat` | `anthropic` SDK | `httpx`, `/v1/chat/completions` |
| Model | `qwen3:8b` (local) | `claude-haiku-4-5-20251001` | `mistral-small-latest` |
| Auth | none (localhost) | `ANTHROPIC_API_KEY` (SDK) | `MISTRAL_API_KEY` (bearer header) |
| `stop_reason` source | **derived** from `tool_calls` presence | explicit field via `_STOP_REASONS` map | `finish_reason` via `_STOP_REASONS` map |
| Tool args on wire | already a dict | dict (`.input`) | **JSON string** both ways (`json.loads`/`dumps`) |
| Tool result correlation | by `tool_name` | by `tool_use_id` | by `tool_call_id` + `name` |
| Tool schema shape | OpenAI envelope | flat (`input_schema`) | OpenAI envelope |

The **same neutral `ToolCall`** serves all three (Ollama reads `.name`, Anthropic reads
`.id`). This is the payoff of carrying the whole `ToolCall` through the round-trip.

Each backend has a `DEFAULT_SYSTEM` prompt constant and a `system_prompt` constructor
arg; `call_model(..., system=...)` allows a per-call override.

---

## 6. Tools & execution

`agentAPI/tools.py`:

- **`run_bash(command: str) -> BashResult`** — `subprocess.run(["bash","-c",cmd],
  shell=False, capture_output=True, check=False, stdin=DEVNULL, text=True, timeout=10)`.
  Deliberate choices: real *bash* (not `/bin/sh`); non-zero exit is information, not an
  exception; `stdin=DEVNULL` prevents stdin-reader hangs; timeout → `BashResult(-9, …,
  timeout=True)`.
- **`BashResult`** (frozen): `returncode, stdout, stderr, timeout`. `.render(max_length=2000)
  -> str` is the **anti-corruption layer between executor and model** — a distinct audience
  from CLI/`logger`. Head/tail truncation with an explicit `…[truncated N chars]…` marker
  (a silent cap is a lie to the model); `(no output)` marker; stderr section only when
  non-empty (its presence is itself a failure signal).
- **Tool registry** — one neutral declaration feeds all three backends:
  ```python
  @dataclass(frozen=True)
  class Tool:
      name: str
      description: str
      parameters: dict[str, Any]      # raw JSON-Schema body (the vendor-identical part)
      func: Callable[..., str]        # returns the model-facing string

  REGISTRY = {t.name: t for t in _TOOLS}   # single source for BOTH declaration & dispatch
  ```
  Each backend renders `REGISTRY.values()` into its own wire schema via `_to_<vendor>_tools`.
  `Agent._execute` dispatches via `REGISTRY[tc.name].func(**tc.arguments)`.

---

## 7. The agent loop

`agentAPI/agent.py` — `Agent(backend, approve=_approve_y_n, max_rounds=10)`:

- **`run(messages: list[Message]) -> ChatResult`** — the I/O-free loop. Call model; while
  `stop_reason == TOOL` and under `max_rounds`: append the `AssistantMessage` (content +
  tool_calls), execute each `ToolCall`, append a `ToolResultMessage`, re-call. This is the
  testable core (driven by a `FakeBackend` in tests).
- **`_execute(tc) -> str`** — returns a string in *all* cases (unknown tool / declined /
  raised exception), so a bad or blocked call becomes a tool-result the model re-plans
  from, never an exception into the loop.
- **`approve: Callable[[ToolCall], bool]`** — the injected **safety gate** (default
  terminal y/n). The whole policy surface is this one callback: human prompt, deny-all
  (`probe_bash.py`), or a future `--auto`/`--dry-run`/`--confirm` selector are all the
  same seam with different policy.
- **`repl()`** — thin terminal driver; owns conversation state (`messages: list[Message]`),
  prints assistant reply + per-turn `tokens_in/tokens_out`.

---

## 8. Observability & testing today

- **Logging**: library modules do `getLogger(__name__)` and emit; the application
  (`__main__.py`) configures once — root floored at `WARNING`, `agentAPI.*` raised to
  `DEBUG` (so vendor `httpcore`/`httpx` noise stays silenced). Each backend `logger.debug`s
  its token counts.
- **Token counts**: surfaced on every `ChatResult` and printed per turn in the REPL.
- **Tests** (45, `pytest`): hand-rolled fakes at every seam (`FakeClient`/`FakeResponse`
  for httpx backends, `FakeMessage` for Anthropic, `FakeBackend` for the loop). Coverage
  includes happy/connection/protocol paths per backend, message & tool round-trip
  translation, the loop's control flow (regression guards for real bugs found), `_execute`'s
  four branches, `run_bash` (incl. bash-not-sh, stdin-no-hang), and `render`/truncation.
- **`probe_bash.py`**: drives the real `Agent.run()` with a **deny-all** policy — records
  the model's tool-intent (first plan + re-plans) without executing anything. Safety is
  structural (`approve` consulted *before* dispatch). Already demonstrated wording-sensitivity
  and run-to-run command variation.

---

## 9. Seams the larger project would build on

These are the load-bearing extension points, already designed for substitution:

1. **`Backend` Protocol** — add providers, or wrap an existing backend (e.g. a recording/
   replaying/cost-accounting decorator) without touching callers. `ChatResult` already
   carries `tokens_in/out` for budget accounting.
2. **`system` parameter on `call_model`** + per-backend `system_prompt` — the natural hook
   for **persona** injection and variation.
3. **`messages: list[Message]`** as the sole context input — the entire **context-management /
   ICL-selection** surface. A context manager / retriever / window-packer would produce this
   list; the neutral `Message` union is what every backend consumes.
4. **Tool `REGISTRY`** — single-source tool declaration; **skills** as tools, or per-run tool
   subsets, slot in here (the noted next step is *injecting* the registry into backends
   instead of module-level import).
5. **`approve` callback** — pluggable execution policy.
6. **`Agent.run()` is I/O-free** — drive it programmatically (as `probe_bash.py` does) for
   batch **A/B runs**, evals, and offline experiments without a terminal.

### Gaps / things a larger system will likely need to add

- **No `GenerationParams`** yet (temperature, max_tokens, top_p, context-length are
  currently fixed per-backend constructor constants) — the designed escape hatch is a
  frozen `GenerationParams` parameter object on `call_model`, added when a *second* common
  per-call knob appears. **A/B testing of decoding params lands here.**
- **No structured run/trace record** — token counts are logged/printed, not persisted.
  Observability of *output quality vs. token budget* needs a trace sink (per call:
  inputs, messages, params, tokens, stop_reason, latency, cost, output, and an eval score).
  `ChatResult` is the natural thing to widen or wrap.
- **No streaming** — all calls are non-streaming.
- **No retries / rate-limit backoff** — Mistral free tier is ~2 RPM; errors translate to
  `BackendError` but aren't retried.
- **No prompt/context budgeting** — nothing trims or summarizes `messages` before a call;
  history grows unbounded.
- **Tool declaration vs. dispatch** are unified now (`REGISTRY`), but tool *selection per
  run* (which subset of skills to advertise) is not yet a concept.
- **Single tool** (`run_bash`); no Python-execution tool yet (was an original goal).
- **Config is hardcoded** (URLs, models, max_tokens in constructors) — no config file/env
  layering beyond API keys.

---

## 10. Running it

```bash
make chat_ollama          # or chat_anthropic / chat_mistral
python -m agentAPI -b ollama
make test                 # pytest -v   (45 tests)
python probe_bash.py "list the files here" -b ollama   # dry-run, nothing executes
```

Env (`.env`, git-ignored): `ANTHROPIC_API_KEY`, `MISTRAL_API_KEY`. Ollama assumes a local
server at `localhost:11434`.

---

## 11. Where to read more

`SESSIONS.md` is an append-only design journal — every decision above has a dated entry
explaining the *why*, the tradeoff considered, and the bug that motivated it. It is the
single best source for understanding the reasoning, not just the result.
