# Context Orchestrator — Build Addendum to Design v2 (code-verified 2026-06-10)

> **What this is.** A pre-build correction layer for `context-orchestrator-design-v2.md`,
> grounded in the *actual* `pyarchAgent` source (`agentAPI/backend.py`, `agentAPI/agent.py`),
> not the summary in `initialPyArchSPECS.md`. v2 is a strong plan; its foundations are sound.
> But four of its statements are either silent on, or wrong about, the substrate as it really
> is — and three of those touch the most expensive-to-change decisions. This addendum records
> the deltas so the build session doesn't discover them at M2/M5.
>
> **How to read it.** Read v2 first (the spec), then this. Sections here tag the v2 text they
> act on: **[SUPERSEDES §x]** = v2 is wrong/incomplete here, this wins; **[CONFIRMS §x]** =
> v2 is right, don't re-litigate. Where v2 and this addendum conflict, **this addendum wins.**
>
> **Verdict.** Safe to build once §A–§D are locked. They touch the two most expensive-to-change
> decisions (the **segment content model** and the **execution primitive**) and the one
> product-defining feature (**per-segment budget legibility**). Settle them *before* M1 freezes
> the segment model and M2 wires the assembler. Everything else in v2 stands.

---

## 1. Substrate facts, verified against the real code

| v2 / spec claim | Status | Evidence |
|---|---|---|
| `call_model(messages, *, system=None) -> ChatResult` | **CONFIRMED**; no `params` yet (ADAPT as v2 plans) | `backend.py:38-43` |
| `ChatResult{stop_reason, content, tokens_in, tokens_out, tool_calls}` | **CONFIRMED** — budget axis is at the boundary | `backend.py:30-36` |
| `ChatResult`/`Message` are **frozen dataclasses**, not pydantic | **CONFIRMED** (matters for §D) | `backend.py:5-36` |
| `Message` union carries tool structure (`AssistantMessage.tool_calls`, `ToolResultMessage.tool_call`) | **CONFIRMED** (matters for §C) | `backend.py:11-23` |
| `Agent.run(messages) -> ChatResult` is the experiment-layer execution engine | **DIVERGES** — see §A | `agent.py:37-52` |
| Error taxonomy / `REGISTRY` / `approve` reusable as-is | **CONFIRMED** | `agent.py:9-35`, `backend.py:45-48` |

The merge is still overwhelmingly additive. But "two small changes to one method signature"
(v2 §10) is an undercount, and the choice of execution primitive is, I think, wrong for the
phase-1 goal. Details below.

---

## A. Execution primitive — drive single-shot `call_model`, not `Agent.run` **[SUPERSEDES §1.1, §5, §10]**

v2 repeatedly routes the experiment layer through `Agent.run` ("exactly as `probe_bash.py`
drives it"). Read what `Agent.run` actually is (`agent.py:37-52`):

- **It mutates the `messages` list you pass in** — appends assistant turns and tool results
  (lines 43, 47), and appends a final `AssistantMessage(result.content)` *even on a no-tool run*
  (line 51). Passing one `window.messages` to N trials means trial 2 starts polluted by trial 1.
  Latent correctness bug in `run_experiment`.
- **It returns only the final `ChatResult`.** Intermediate rounds' `tokens_out` and the repeated
  prefills are discarded; `tokens_in` is the last (largest) prompt, not Σ over rounds. For a
  project whose *entire point* is token-budget observability, recording the single returned
  `ChatResult` of a multi-round run **systematically under-reports its budget.**
- **Tool outputs (`run_bash`) are environment-dependent and the round count varies.** Two trials
  of the *same* assembly can diverge wildly in tokens — so the budget you attribute to the
  assembly is dominated by tool-loop variance, not by the composition you're A/B-ing. This
  **confounds the core experiment.**

**Decision (locked):** the phase-1 experiment layer drives **single-shot
`backend.call_model(messages, system=, params=)`**. `Agent.run` (the tool loop) is deferred to
the orchestration/HITL phase, where emergent working memory *is* the object of study.

Why single-shot is the right phase-1 primitive: `tokens_in` then equals `BudgetReport.total`
(directly reconcilable), the window is reproducible, variance is sampling-only, there is no
list mutation, no lost per-round accounting, and no tool-output confound. It also makes §C
(text-only segments) fully coherent, because a single-shot run generates no tool working memory.

```python
def run_assembly(spec, backend) -> Run:
    window = assembler.assemble(spec)            # (system, messages, BudgetReport)
    result = backend.call_model(                 # SINGLE-SHOT — not Agent.run
        window.messages,
        system=window.system,
        params=spec.params,
    )
    # result.tokens_in is now directly reconcilable with window.budget.total_tokens
    return Run(..., tokens_in=result.tokens_in, tokens_out=result.tokens_out, ...)
```

**If/when agentic A/B enters scope (deferred phase),** `Agent.run` must first gain, and the
build session must not forget:
```python
#  (a) cumulative usage across rounds — sum tokens_in/out, or return a per-round trace;
#      the single returned ChatResult is NOT the run's budget.
#  (b) no input-list aliasing across trials — assemble fresh / deep-copy per trial,
#      because run() mutates its argument (agent.py:43,47,51).
#  (c) system + params threaded at BOTH call sites (agent.py:38 and :49) — today neither
#      is passed.
```

**§10 edit:** remove `Agent.run` from the phase-1 `[REUSE]` critical path; it is a deferred-phase
engine with the three prerequisites above. *Affected: M2 (`run_assembly`), M5 (`run_experiment`).*

---

## B. Per-segment budget needs a pre-call tokenizer — it cannot be "reported post-call" **[SUPERSEDES §4.1, §8]**

v2 offers a "leaner first pass: report post-call from real `ChatResult` usage; enforce later."
This is incompatible with the product's headline feature. `ChatResult.tokens_in` is **one
aggregate integer** (`backend.py:34`). The per-`SegmentType` stacked bar (screen 2, "the heart"),
per-segment attribution (screen 3, `BudgetReport.per_segment` §4.2, `VariantRow.per_type_tokens`
§7.2, `RunDetailView` per-message tokens §7.2) all require **decomposing the window per segment**
— which is *only* possible by tokenizing each segment **pre-call**. A single post-call total
cannot be decomposed. And `SegmentVersion.token_estimate` is nullable (§2), so it can't be the
sole basis either.

**Decision (locked):** per-segment budget is computed by a **pre-call tokenizer**, available to
the Assembler as a real `Backend` capability. The total is **reconciled** against
`ChatResult.tokens_in` post-call, and the estimate/actual delta is surfaced.

```python
class Backend(Protocol):
    def call_model(self, messages, *, system=None,
                   params: GenerationParams) -> ChatResult: ...
    def count_tokens(self, text: str) -> int: ...   # [NEW] — NOT optional; budget legibility
```

Notes for the builder:
- Tokenization is **not additive across segment boundaries**, so `per_segment` sums are
  *independent-tokenization estimates*; document that the sum may differ from `total_tokens` by
  boundary effects. `ChatResult.tokens_in` is the authoritative **total**; per-segment is the
  legible-but-approximate breakdown.
- **Pick the tokenizer at M1.** Pragmatic default: one shared local tokenizer (e.g. the model's
  HF tokenizer for Ollama/qwen) used uniformly, accepting small per-backend inexactness, rather
  than chasing Anthropic's count-tokens endpoint + Mistral's tokenizer for exactness no screen
  needs. Decide explicitly; don't leave it to M3 when the dashboard suddenly needs it.
- `SegmentVersion.token_estimate` stays advisory/cache, never the budget source of truth.

*Affected: M1 (pick tokenizer), M2 (assembler attribution + `Backend.count_tokens` on 3 backends).*

---

## C. Phase-1 segments are text-only **[SUPERSEDES §2 "working memory may replay assistant turns"]**

The `Message` union carries structured tool data: `AssistantMessage(content, tool_calls)` and
`ToolResultMessage(tool_call: ToolCall, content)` (`backend.py:11-23`). A segment modeled as
`content: str` + `role_hint: str` **cannot reconstruct either.** v2 §2's "working memory may
replay assistant turns" therefore over-promises relative to its own type.

**Decision (locked):** phase-1 segments are **text-only**.
- `role_hint ∈ {"user", "assistant"}`, default `"user"`. TASK_SPEC / KNOWLEDGE / SKILL → user;
  WORKING_MEMORY may be assistant-role *text*.
- **Tool-call / tool-result structure is out of segment scope.** Any agentic working memory
  generated *during* a run (deferred phase, when `Agent.run` is used) is captured in the run
  **snapshot** (display-only, §6), never authored as a replayable segment.
- Structured-content segments are a *later* typed-content extension behind the same
  immutable-version model — designed-around, not built now.

This is the foundational one: M1 is where the segment model locks and "most of the
abstraction-design learning lives" (v2 §9). It also reinforces §A — single-shot runs generate no
tool working memory, so text-only segments are fully sufficient for phase 1.

**§2 edit:** replace the "replay assistant turns" sentence with the scope statement above; fix
`role_hint` domain to `{user, assistant}` and document the `user` default. *Affected: M1.*

---

## D. Adaptation checklist — corrected and expanded **[SUPERSEDES §10]**

v2 §10 says "two small, already-anticipated changes to one method signature." The real surface:

1. **[ADAPT] `Backend.call_model` gains `params: GenerationParams`** — Protocol (`backend.py:38`)
   + 3 backend impls. *(As v2 plans.)*
2. **[ADAPT] `Backend.count_tokens(text) -> int`** — real, not optional (§B). 3 impls.
3. **[DEFER] `Agent.run` gains `system` + `params` threading at two call sites** (`agent.py:38,
   :49`) plus cumulative usage + no list aliasing (§A). *Phase-1 single-shot avoids this entirely;
   record it as the deferred-phase prerequisite so it isn't missed later.*
4. **[NEW] pydantic ↔ frozen-dataclass + `Message`-union JSON serialization.** `domain/` is
   pydantic; `Message`/`ChatResult` are stdlib **frozen dataclasses** to be *reused*, not
   redefined (v2 §6). So `AssembledWindow.messages` and the run snapshot must serialize a union
   whose members **have no type discriminator** and where `ToolResultMessage` nests a `ToolCall`.
   Decide at M1: a tagged JSON representation (add a `"role"`/`"kind"` tag on
   serialize/deserialize) + pydantic config to embed/validate the dataclasses
   (`arbitrary_types_allowed` or a custom (de)serializer). This is the *first* thing you hit in M1.

*Affected: M1 (#4), M2 (#1, #2). #3 is deferred-phase.*

---

## E. Cheaper decisions — low cost, decide consciously

- **System-channel ownership.** The substrate has *three* system sources: each backend's
  `DEFAULT_SYSTEM`, the `system_prompt` ctor arg, and the per-call `system=`. The orchestrator
  must always pass the assembled persona via `call_model(system=...)` **and** ensure the backend
  default doesn't leak — otherwise `BudgetReport.system_tokens` attribution lies.
- **`Run.score` → `Run.scores: list[Score]`.** Budget-vs-quality usually means several signals
  (human + exact-match + latency). By v2's *own* "cheap insurance now" logic (§11.3's `source` /
  nullable `assembly_id`), make it a list now; `VariantRow` aggregates per `score_kind`.
- **§2 overclaim:** "adding a `SegmentType` requires no schema changes elsewhere" is true for
  *storage* but false for *behavior* — the assembler routes by type (§4.2) and
  `per_type_tokens` (§7.2) is keyed by it. Adding a type is cheap, not free. Reword.
- **Scope honesty on the unspecified parts.** The dashboard *is* the product, yet v2 leaves the
  frontend "bought/generated" and the HTTP layer "hand-rolled, no framework." A thin framework
  (Flask/FastAPI) behind the same `ProjectionStore` + write-service seam is architecturally free
  and reclaims time for the parts that *are* the point (assembler, budget, experiment). Reconsider
  before M3. And keep **§11 companion mode** to its two schema hooks (`source`, nullable
  `assembly_id`) only — don't let the section's detail pull the proxy forward.

---

## F. Locked and good — do not re-litigate **[CONFIRMS v2]**

These are the expensive-to-get-wrong decisions v2 got right; build on them without second-guessing:

- Reproducible-by-reference assemblies (refs + params + `policy_id`; never snapshot opaque text).
- `policy_id` stamped despite one policy; `GenerationParams` folded in *now*, not retrofitted.
- Append-only truth + rebuild-on-read projections behind `ProjectionStore`, with the named
  UI-driven migration trigger to update-on-write.
- Single-variable A/B discipline with `Experiment.varied_dimension` spanning both axes
  (segment *or* param).
- `Evaluator` port as a quarantine for the unbounded relevance/quality research problem.
- Reuse of the neutral `Message` / `ChatResult` rather than redefining them.
- Vertical-slice milestones (always something working).

---

## G. Decisions this addendum locks (the build checklist)

| # | Decision | Choice | Corrects | Milestone |
|---|---|---|---|---|
| A | Phase-1 execution primitive | **Single-shot `call_model`**; `Agent.run` deferred (with 3 prereqs) | §1.1, §5, §10 | M2 / M5 |
| B | Per-segment budget source | **Pre-call tokenizer** (`Backend.count_tokens`); reconcile total vs `ChatResult` | §4.1, §8 | M1 / M2 |
| C | Segment content model | **Text-only**; `role_hint ∈ {user, assistant}`; tool structure out of scope / snapshot-only | §2 | M1 |
| D | Adaptation surface | `call_model+=params`, `+count_tokens`, **+pydantic↔dataclass+union-JSON**; `Agent.run` threading deferred | §10 | M1 / M2 |
| E1 | System channel | Orchestrator owns `system=` per call; backend default must not leak | — | M2 |
| E2 | Score cardinality | `Run.scores: list[Score]` | §5, §7.2 | M1 |
| E3 | Frontend/HTTP | Reconsider thin framework behind the same seams | §7.3-7.4 | M3 |

None of these threaten the architecture; all are cheaper to settle now than after M1 freezes the
segment model and M2 wires the assembler. Lock A–D, note E, and start building.

---

*Evidence base: `pyarchAgent/agentAPI/backend.py` (49 lines) and `agentAPI/agent.py` (77 lines),
read 2026-06-10. Re-verify against those files if the substrate has moved since.*
