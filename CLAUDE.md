# Context Orchestrator — Instructions for Claude

## What this is

A context-orchestration system — typed **segments** → reproducible **assemblies** → recorded
**runs** — for A/B testing and budget-vs-quality observability of LLM context. Built on the
sibling **`pyarchAgent`** substrate. Like pyarchAgent, it's a real system built as deliberate
engineering practice: the craft is as much the point as the product.

## Read before building (the architecture is already specced)

1. `context-orchestrator-design-v2.md` — the design.
2. `context-orchestrator-design-v2-addendum.md` — code-verified corrections. **On any conflict,
   the addendum wins.** Four decisions are locked there: single-shot `call_model` execution
   (not `Agent.run`), a pre-call tokenizer for per-segment budget, text-only segments, and the
   expanded adaptation list.
3. `initialPyArchSPECS.md` — the substrate. Its `Backend` / `Message` / `ChatResult` / `Agent`
   are **reused, not rebuilt**.

Don't re-open the locked decisions or v2's "locked & good" list without a real reason.

## Work mode — hybrid by layer

**You author the load-bearing design; I implement the mechanical breadth.**

- **Yours** (where the craft lives): domain types, the ports (`Repository` / `Assembler` /
  `Evaluator`), the `concat-v1` assembler, DB schema, read-model shapes, endpoint contracts —
  the abstraction decisions.
- **Mine, on request**: vendor adapters, SQLite row↔pydantic mapping bodies, HTTP
  serving/routing, test scaffolds, repetitive CRUD — the breadth that would otherwise crowd out
  the design work.
- Ask "how do I X?" → I explain the *shape* and the tradeoff so you write the core. Ask me to
  build a mechanical layer → I write it against the agreed design.
- Push back on shaky designs; surface the tradeoff; don't rubber-stamp. Mention the sharper
  idiom *after* it works, not before. One abstraction at a time; the M1→M5 roadmap emerges from
  the work.

**Craft in focus: all four** — Python design & abstraction, systems & data-flow, persistence/SQL,
API & dashboard. The same split holds across every one: you own the design-bearing decision, I
handle its mechanical realization (e.g. SQL — you design the schema and the row↔model *approach*,
that's the exercise; I write the boilerplate). Git hygiene carried over.

## Scope discipline

- Build the **observe → re-run** spine first. Design *around* the deferred items (variable
  assembly policy, automated relevance/quality scoring, streaming, retries, companion/proxy mode)
  — not *for* them.
- Keep the research risk contained behind the `Evaluator` seam, dumb-at-first. Don't let
  companion mode smuggle the structure-recovery problem back in.
- No premature frameworks; reuse the substrate rather than re-deriving it.
  Reproducibility-by-reference is non-negotiable.

## Security posture

Relaxed but proportionate (from pyarchAgent): low-privilege account, human `approve` on tool
calls. Add sandboxing / allowlists when unattended execution or untrusted input becomes real —
note risks when relevant, don't gate progress on them.

## Session log & git

- `SESSIONS.md` (repo root), pyarchAgent convention: dated `### YYYY-MM-DD — <topic>` entries at
  checkpoints — what was built, what was decided and why, what was hard, what's next.
  Append-only; I maintain it unasked.
- Git repo on `main`; commit small and often, clear messages. **Substrate vendored by copy**
  (2026-06-10): `agentAPI/` is treated as upstream from the sibling `pyarchAgent` repo — re-sync
  by re-copying, and keep any edits to it minimal and explained.
