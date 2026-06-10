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