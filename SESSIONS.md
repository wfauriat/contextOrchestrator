# Context Orchestrator — Sessions

Append-only design journal (pyarchAgent convention). Newest entries at the bottom.

### 2026-06-10 — Vendor the pyarchAgent substrate; green baseline

- **Built:** copied the `pyarchAgent` substrate into this repo by **copy** — `agentAPI/` (8
  modules: `backend`, `agent`, `tools`, the three backends, `__init__`, `__main__`) + its 5-file
  test suite, plus `pyproject.toml` (pytest config), `Makefile`, `.gitignore`. No edits to any
  `agentAPI` source.
- **Verified:** all **45 tests pass** from the new repo on the existing venv (`/opt/venvs/pyDS`).
  Known-good vendored baseline.
- **Decided — vendoring = copy** (over submodule/import): keeps the substrate diffable against
  upstream `pyarchAgent` and lets us re-sync by re-copying; cost is manual re-sync, accepted.
  Recorded in `CLAUDE.md`.
- **Decided — defer contract edits to M2:** the warranted adaptations from the design addendum
  (`Backend.call_model += params: GenerationParams`, `+ count_tokens`) depend on domain types /
  a tokenizer that don't exist yet and would break the 45 tests. Land a clean baseline first,
  then adapt deliberately with their own tests. (See the addendum §A, §B, §D.)
- **Hard / open:** layout for the *new* code. Keep `agentAPI` name (treat as upstream). Open
  fork: flat top-level packages (design §6) vs. one umbrella package beside `agentAPI/` — decide
  before M1 creates `domain/`.
- **Next:** M1 — `domain/` types + `Repository` port + sqlite adapter, tested in the substrate's
  hand-rolled-fake style. Resolve the layout fork first.

### 2026-06-11 — M1 domain types + concat-v1 assembler; end-to-end spike fires

- **Built (`orchestrator/`):** the domain layer for the observe spine —
  `domain.py`: `SegmentType` (5-type routing enum), `SegmentVersion` (frozen, immutable
  content+metadata), `Segment` (identity), `SegmentRef` (frozen pin), `ResolvedSegment`
  (the `Segment.type` + `SegmentVersion` join the resolver returns), `GenerationParams`
  (frozen, intersection-of-backends), `AssemblySpec` (ordered refs + params + `policy_id`).
  `assembly.py`: `AssembledWindow` (plain `@dataclass`, holds `(system, messages)`) +
  `ConcatV1Assembler` — PERSONA → joined `system`, everything else → ordered `Message`s by
  `role_hint`. Plus `scratch_assembly.py`, a throwaway harness.
- **Verified:** full chain fires end-to-end through `OllamaBackend` (qwen3:8b) —
  `segments → AssemblySpec → concat-v1 → (system, messages) → call_model → ChatResult`.
  Two PERSONA segments routed to `system` visibly shaped output (pirate voice); real
  `tokens_in=176` observed. Routing-as-design confirmed in behavior, not just structure.
- **Decided — layout fork resolved:** umbrella package `orchestrator/` beside `agentAPI/`
  (flat modules within it for now — `domain.py`, `assembly.py` — promote to subpackages
  under pressure), not flat top-level packages. Closes the open fork from 2026-06-10.
- **Decided — spike sidesteps addendum #D:** `AssembledWindow` is a plain dataclass, not
  pydantic, so the pydantic↔frozen-dataclass `Message`-union JSON problem is deferred to
  when we *persist* a window (Run snapshot), not paid for an in-memory spike.
- **Decided — resolver-callable seam:** assembler depends on
  `Callable[[SegmentRef], ResolvedSegment]`; in-memory dict now, `Repository` at M2, same
  interface. Assembler never changes.
- **Held to discipline (carried-but-not-wired):** `params` recorded on `AssemblySpec` but not
  threaded through `call_model` (M2 ADAPT-1); no `count_tokens`/`BudgetReport` yet (M2,
  addendum B); single-shot `call_model`, not `Agent.run` (addendum A); segments text-only
  (addendum C). `policy_id` stamped though only one policy exists.
- **Surfaced:** the system-leak risk (addendum E1) is live — backends fall back to a hardcoded
  `DEFAULT_SYSTEM` when `system is None`; safe here only because personas make it non-`None`.
  Handle the no-persona case deliberately at M2 or budget attribution lies.
- **Next:** finish M1 properly — design the sqlite schema + `Repository` port + row↔pydantic
  mapping, and tackle the #D serialization seam (tagged union JSON) since persistence forces it.