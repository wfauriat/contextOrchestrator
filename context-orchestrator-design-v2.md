# Context Orchestrator — Design Reference v2 (reconciled with `pyarchAgent`)

> **What changed from v1.** v1 was a greenfield sketch that modeled the assembled context
> window as a single string and treated persona as just another segment. You already have a
> working substrate — `pyarchAgent`, a vendor-neutral agent wrapper with the hardest seam
> (model port) already built and tested. This v2 folds that reality in: the window is a
> structured `list[Message]` + a separate `system` channel, the existing `Backend` Protocol
> *is* the model port, and `GenerationParams` enters as a first-class experiment variable. The
> plan is no longer idealized-on-paper; it describes the system you will actually build on top
> of code that exists.
>
> **Scope lock (phase 1).** Typed **segments** (stable IDs, immutable versions); an **assembly**
> as an ordered set of segment-version references + a decoding-param set, stamped with a
> constant `policy_id`, deterministically reconstructable; the **ports** (model = existing
> `Backend`; repository; assembler; evaluator); an **experiment layer** varying *segment
> content and/or decoding params*; and an **observe → re-run** data flow with read models and a
> JSON API.
>
> **Deliberately deferred** (designed *around*, not *for*): variable assembly policies, live
> between-inference-step control, automated relevance scoring, automated quality judgment,
> streaming, retries.
>
> **How to read this.** Interfaces and contracts only — signatures + docstrings describe the
> *what*; bodies and hard decisions are yours. Invariants are stated in prose so you implement
> them consciously. Sections tagged **[EXISTS]** map onto code already in `pyarchAgent`;
> **[ADAPT]** means existing code needs a small, named change to fit; **[NEW]** is net-new.

---

## 0. The one mental model

Everything that reaches the model is a **budgeted resource competing for the same space**.
Persona, skills/exemplars, knowledge, task spec, and working memory are *typed segments* with
different lifetimes and relevance profiles. The system:

1. **assembles** typed segments into a concrete window — a `system` string + an ordered
   `list[Message]` — and reports the budget, then
2. **attributes** an outcome + quality back to the exact assembly (and decoding params) that
   produced it.

The **assembly** — not the run — is the unit of analysis. A run is one execution; an assembly
is the reproducible composition (segment versions + decoding params + policy) that produced it.
A/B testing means holding the task fixed, varying one thing (a segment version *or* a decoding
param), and comparing outcomes across assemblies.

The spine constraint: **an assembly must be fully reconstructable from stored references** —
segment IDs + versions + params + `policy_id` — never snapshotted as opaque text.
Reproducibility is non-negotiable because every fair comparison depends on it.

---

## 1. The existing substrate — `pyarchAgent` (as of 2026-06-10)

A from-scratch, multi-backend LLM agent wrapper built as a deliberate engineering exercise.
What matters here is that three of v1's load-bearing abstractions **already exist, tested**,
and the project's discipline (neutral domain types, injection seams, I/O-free loop) is exactly
what this larger scope assumes. It is a substrate to build on, not a framework to fight.

**Stack:** Python 3.12, `httpx` (Ollama/Mistral raw HTTP), `anthropic` SDK, `python-dotenv`,
`pytest`, `mypy`/`pyright`. 45 tests passing. Append-only design journal in `SESSIONS.md`.

### 1.1 What carries over directly **[EXISTS]**

- **Neutral message model** — a discriminated union (`UserMessage | AssistantMessage |
  ToolResultMessage`) with frozen dataclasses. `ToolResultMessage` carries the whole `ToolCall`
  so id/name can't drift. *This is the assembler's output element type* — the thing every
  backend consumes. v1's fictional flat-string window is replaced by this real structure.
- **`ChatResult`** — `stop_reason, content, tokens_in, tokens_out, tool_calls`. The
  `tokens_in/tokens_out` fields mean **the budget axis is already surfaced at the boundary**,
  per call, per backend. v1's `TokenUsage` is these two fields.
- **`Backend` Protocol** — `call_model(messages, *, system=None) -> ChatResult`. Structural
  typing (not ABC) deliberately covers both the backend↔backend seam and the backend↔SDK seam.
  **This is v1's model port, already insulated** — vendor SDKs live at the edges; the middle
  speaks one vocabulary. Three adapters exist (Ollama, Anthropic, Mistral).
- **Error taxonomy** — `BackendError` → `Connection / Response / Contract`. Types are
  vendor-agnostic; messages are vendor-specific. Reusable as-is.
- **I/O-free `Agent.run(messages) -> ChatResult`** — the tool loop with no terminal I/O,
  already driven programmatically by `probe_bash.py`. **This is what the experiment layer drives
  for batch A/B runs.**
- **Tool `REGISTRY`** — single-source declaration + dispatch; each backend renders it to its own
  wire schema. The seam where **skills-as-tools / per-run tool subsets** will live.
- **`approve` callback** — pluggable execution policy (human y/n, deny-all, future auto/dry-run).

### 1.2 Named gaps this project fills (from the substrate's own spec)

- No `GenerationParams` (temperature/max_tokens/top_p fixed per backend) — **the decoding-param
  A/B axis lands here** (§5).
- No persisted run/trace record — token counts are logged, not stored. **The `Run` record is
  this trace sink** (§6).
- No prompt/context budgeting — history grows unbounded. **The assembler + budget report
  address this** (§4).
- Tool *selection per run* (which subset to advertise) is not yet a concept. **Becomes a
  `SKILL`-type segment** (deferred mechanism, §4 note).
- Config hardcoded; no streaming; no retries. Deferred, aligned with phase-1 scope.

---

## 2. Segment & version types **[NEW]**

A **segment** is a named, typed container owning a history of immutable versions. The segment
is stable identity; a **segment version** is a frozen content+metadata snapshot. Editing =
appending a new version, never mutating.

### Invariants (prose; enforce as you author)
- **Stable ID** for life. **Immutable versions** once written. **Monotonic `version_no`** within
  a segment. **Provenance** recorded on every version (hand-authored / derived / imported).

```python
from enum import Enum
from datetime import datetime
from pydantic import BaseModel


class SegmentType(str, Enum):
    """Category tag the assembler reads to decide routing & ordering. Carries no behavior.
    NOTE the routing consequence (see §4): PERSONA is routed to the model's `system`
    channel, NOT inlined into the message list — because the existing Backend keeps system
    separate, exactly mirroring how providers model it."""
    PERSONA = "persona"          # -> routed to the `system` channel
    SKILL = "skill"              # ICL exemplars / (later) tool-subset selection
    KNOWLEDGE = "knowledge"      # retrieved / reference material
    TASK_SPEC = "task_spec"      # the concrete instruction for this run
    WORKING_MEMORY = "working_memory"  # history / scratch / evolving state
    # Adding a type must not require schema changes elsewhere.


class SegmentVersion(BaseModel):
    """Immutable snapshot of a segment's content + metadata.

    Contract:
      - Frozen after creation; append a new version to 'edit'.
      - `token_estimate` is advisory; the authoritative count comes from the model port
        at assembly time (see §4 / §1.1 — backends already report real token usage).
      - `derived_from` records provenance (None if hand-authored).
      - `role_hint` lets a non-persona segment declare how it becomes a Message (user vs
        assistant vs a tool-result). Most segments are user-role context; working memory may
        replay assistant turns. Decide the default and document it.
    """
    segment_id: str
    version_no: int
    content: str
    token_estimate: int | None
    derived_from: tuple[str, int] | None
    role_hint: str | None              # e.g. "user" | "assistant"; None -> policy default
    created_at: datetime
    # meta: loose dict now (documented per-type convention); tighten to typed later
    # without touching segment identity.


class Segment(BaseModel):
    """Stable identity + typed category. Content lives in versions, not here.

    Contract: `id` stable for life; `latest_version_no` kept consistent by the repository
    (decide: store denormalized, or always derive)."""
    id: str
    type: SegmentType
    name: str
    created_at: datetime
    latest_version_no: int | None
```

> **Reconciliation note.** v1 had no `role_hint`, because it modeled the window as a flat
> string where role didn't exist. The substrate's `Message` union *requires* a role, so a
> non-persona segment must know which `Message` subtype it becomes. This is the price of the
> (better) structured window — and a small one.

---

## 3. The Assembly — first-class, reproducible **[ADAPT of v1]**

An **assembly** is the unit A/B and budget attach to: an ordered list of segment-version
references, **plus a decoding-param set**, plus the `policy_id` constant. It is NOT the
assembled messages — those are *derived* from the assembly, deterministically.

### Invariants
- **Reconstructable**: spec + immutable versions + params + named policy → the exact same
  `(system, messages)`. Determinism is mandatory; it's what makes a comparison fair.
- **Reference, don't embed**: store `(segment_id, version_no)` pairs and a params reference,
  never resolved text. (Cache resolved text in a run snapshot for display — §6.)
- **Stamp the policy** even though there's one in v1 (`"concat-v1"`). One field; cheap
  insurance that keeps "vary the policy later" an addition, not a migration.

```python
class SegmentRef(BaseModel):
    """A pin to one immutable segment version. Pinning (not 'latest') is the point —
    it's what makes the assembly reproducible."""
    segment_id: str
    version_no: int


class GenerationParams(BaseModel):
    """The decoding-param set — frozen. This is the substrate's named-but-unbuilt escape
    hatch (temperature/max_tokens/top_p were fixed per-backend constants) promoted to a
    first-class, varyable experiment input. Holding this fixed while varying a segment, OR
    fixing segments while varying this, are both valid A/B axes.

    Contract:
      - Treat as the INTERSECTION of backend capabilities, never the union (the substrate's
        Protocol discipline). Knobs every backend honors; provider-specific extras stay out.
      - Frozen + value-equal, so two assemblies with identical params are comparable.
    """
    max_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    stop: tuple[str, ...] = ()


class AssemblySpec(BaseModel):
    """The reproducible recipe for a run's context + decoding.

    Contract:
      - `ordered_refs` is the exact order the policy consumes (concat-v1 = this order).
      - `params` is the decoding-param set for the run.
      - `policy_id` names the assembly policy; constant in v1.
      - Identical (ordered_refs, params, policy_id) MUST assemble + decode identically.
        Hash these three for a stable assembly identity (dedup / 'run this before?').
    """
    id: str
    ordered_refs: list[SegmentRef]
    params: GenerationParams
    policy_id: str
    created_at: datetime
```

> **Reconciliation note.** v1's `AssemblySpec` had no `params`. Folding `GenerationParams` in
> *now* is the one change worth doing up front: "A/B on temperature" is a feature you explicitly
> want, and the substrate already identified `GenerationParams` as its designed escape hatch.
> Retrofitting it into the experiment layer later is the annoying kind of change; adding the
> field now is free.

---

## 4. The ports

Ports are the seams you own; vendor SDKs live strictly inside adapters. Two of the four ports
already exist in the substrate.

### 4.1 Model port — **[EXISTS as `Backend`]**

Do not rebuild this. The substrate's `Backend` Protocol *is* the model port, already insulated
and tested across three providers. Two adaptations only:

```python
# EXISTING (substrate), unchanged in spirit:
class Backend(Protocol):
    def call_model(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> ChatResult: ...

# ADAPT 1: thread GenerationParams through. The substrate's spec already names this as the
# 'second common per-call knob' trigger for the params object. Add it as a keyword:
#     def call_model(self, messages, *, system=None, params: GenerationParams) -> ChatResult
# Each backend maps params -> its wire form; absent knobs fall back to its constant defaults.
#
# ADAPT 2 (deferred mechanism, design the seam now): per-call tool subset, so a run can
# advertise a chosen subset of REGISTRY rather than all of it:
#     ... tools: list[str] | None = None   # names from REGISTRY; None -> all
# This is the hook SKILL-type segments will drive. Build it when you do the registry-injection
# refactor the substrate already flagged.
```

`ChatResult.tokens_in/tokens_out` remain the authoritative budget signal (§6 widens/wraps it
into the `Run` trace). The error taxonomy is reused unchanged.

> **Token counting for budgeting.** v1 put `count_tokens` on the model port. The substrate gets
> real post-hoc counts from `ChatResult`, which covers *run* accounting. For *pre-assembly*
> budgeting (will this fit before I call?), add a `count_tokens(text) -> int` to the Backend
> Protocol as an optional capability, OR accept that v1's assembler can budget on the advisory
> `token_estimate` + reconcile against the real `ChatResult` after the call. Decide based on
> whether you want the assembler to *enforce* budget pre-call or merely *report* it post-call.
> Leaner first pass: report post-call from real usage; enforce later.

### 4.2 Assembler port — the heart **[NEW, targets the real window shape]**

Turns an `AssemblySpec` into a concrete window **and** a budget report. The window is the
substrate's real shape: a `system` string + an ordered `list[Message]`.

```python
from abc import ABC, abstractmethod


class BudgetReport(BaseModel):
    """What the assembler reports about the window it built. Budget axis lives here.
    Attribution spans BOTH channels (system + each message)."""
    total_tokens: int
    per_segment: list[tuple[SegmentRef, int]]   # tokens attributed to each segment
    system_tokens: int                          # the persona/system channel's cost
    truncated: bool


class AssembledWindow(BaseModel):
    """The concrete inputs to Backend.call_model — the real shape, not a flat string."""
    system: str | None           # routed from the PERSONA segment(s)
    messages: list[Message]      # ordered, from the non-persona segments via role_hint
    budget: BudgetReport


class Assembler(ABC):
    """Deterministic: same (spec, resolved versions, policy) -> same AssembledWindow.

    concat-v1 contract:
      - resolve each SegmentRef to its immutable version,
      - route PERSONA-type segments to `system` (joined with a fixed, documented separator),
      - turn each non-persona segment into a Message via its role_hint (default: UserMessage),
        in ordered_refs order, into `messages`,
      - attribute tokens per segment (and system_tokens) for the BudgetReport,
      - over-budget policy: ERROR rather than silently truncate (truncation is itself a
        deferred policy concern; erroring surfaces the problem honestly).
    """

    @abstractmethod
    def assemble(self, spec: AssemblySpec) -> AssembledWindow: ...
```

> **Reconciliation note — the central change.** v1: `assemble(spec) -> {text: str}`. v2:
> `assemble(spec) -> {system, messages, budget}`. This is strictly better and forced by the
> substrate: providers consume structured messages + a separate system, so the assembler should
> target that, not a fictional pre-flattened blob. Persona-as-`system` also validates v1's claim
> that persona is "the beginning of the context window" — so much so that the API gives it its
> own field.

### 4.3 Repository port — persistence behind an interface **[NEW]**

`sqlite3` (stdlib) is plenty for phase 1; keep it behind this interface so Postgres is a local
swap. Map rows ↔ pydantic yourself (no ORM) — that mapping is an abstraction exercise, not a
dependency. Append-only where the domain is (versions, runs).

```python
class Repository(ABC):
    # segments & versions
    @abstractmethod
    def create_segment(self, segment: Segment) -> Segment: ...
    @abstractmethod
    def append_version(self, version: SegmentVersion) -> SegmentVersion:
        """Reject version_no != latest+1 (monotonicity); update latest_version_no."""
    @abstractmethod
    def get_version(self, segment_id: str, version_no: int) -> SegmentVersion: ...
    @abstractmethod
    def list_segments(self, type: SegmentType | None = None) -> list[Segment]: ...
    # assemblies (incl. their GenerationParams)
    @abstractmethod
    def save_assembly(self, spec: AssemblySpec) -> AssemblySpec: ...
    @abstractmethod
    def get_assembly(self, assembly_id: str) -> AssemblySpec: ...
    # runs & snapshots
    @abstractmethod
    def save_run(self, run: "Run") -> "Run": ...
    @abstractmethod
    def get_run(self, run_id: str) -> "Run": ...
    @abstractmethod
    def list_runs(self, assembly_id: str | None = None) -> list["Run"]: ...
```

### 4.4 Evaluator port — the contained research-risk seam **[NEW]**

Quality/effect is where research risk tries to creep back. Contain it like the
context-intelligence: pluggable, dumb-at-first. v0 = a manual human score or a trivial metric.
The A/B *machinery* (§5) is pure engineering; the *judgment* plugs in here.

```python
class Score(BaseModel):
    value: float                 # higher = better, by convention
    kind: str                    # "human" | "exact_match" | "contains" | "latency_inv" | ...
    note: str | None = None


class Evaluator(ABC):
    """Run -> Score. v0: ManualEvaluator (records a human score) or a trivial metric.
    The LLM-judge / relevance scorer is a LATER impl behind this same interface."""
    @abstractmethod
    def score(self, run: "Run") -> Score: ...
```

---

## 5. The experiment layer — observability meets A/B **[NEW, drives existing `Agent.run`]**

Thin orchestration over the ports. Holds a **task fixed**, varies **one thing** (a segment
version *or* a `GenerationParams` knob), runs **N trials**, records `(assembly, result, budget,
score)` tuples. No framework — and crucially, it **drives the substrate's I/O-free `Agent.run`**
for execution, exactly as `probe_bash.py` already drives it.

```python
class Run(BaseModel):
    """One execution of one assembly. The atomic observability record AND the trace sink the
    substrate currently lacks. It widens/wraps ChatResult with everything needed to compare.

    Contract:
      - References AssemblySpec by id (reproducible) + a resolved snapshot for display (§6).
      - Captures real usage from ChatResult (authoritative budget), the params used, latency,
        stop_reason, and optionally a Score.
      - Append-only: never edited after completion.
    """
    id: str
    assembly_id: str
    completion: str
    tokens_in: int               # from ChatResult — the substrate already surfaces these
    tokens_out: int
    stop_reason: str             # from the substrate's StopReason enum
    latency_ms: int | None
    budget: BudgetReport
    score: Score | None          # filled by an Evaluator, possibly after the fact
    created_at: datetime


class Experiment(BaseModel):
    """Fixed task + variants differing in exactly one dimension.

    Contract:
      - All variants share the same TASK_SPEC ref (the held-fixed part).
      - They differ in exactly one thing: a varied SegmentRef OR a varied params knob.
        `varied_dimension` records which, so the dashboard labels the axis correctly.
    """
    id: str
    name: str
    held_fixed: list[SegmentRef]
    varied_dimension: str            # e.g. "segment:persona" | "param:temperature"
    assembly_ids: list[str]
    created_at: datetime
```

Orchestration you'll author over the ports + `Agent.run`:
- `run_assembly(spec) -> Run` — assemble → `Agent.run(window.messages, system=...)` via the
  chosen backend → record real usage/latency. The base unit.
- `run_experiment(experiment, trials_per_variant) -> list[Run]` — variants × trials, each a Run.
- `compare(experiment) -> ExperimentGrid` — aggregate runs into per-variant budget + score
  (the §7 read model). Observability and A/B in one.

> **Reconciliation note.** v1 varied "segment content only." v2 varies *segment content OR a
> decoding param*, because `GenerationParams` is now first-class. `Experiment.varied_dimension`
> captures which, keeping the single-variable discipline (change one thing, attribute the
> effect) while supporting both axes.

---

## 6. Phase-1 data flow, snapshots, module boundaries

Flow, observe-first:

```
author/edit segments         -> Repository (segments, versions)
build an AssemblySpec         -> Repository (assemblies + params)   [references, not text]
run_assembly(spec):
    Assembler.assemble(spec)  -> AssembledWindow (system, messages, BudgetReport)
    Backend.call_model(messages, system=, params=)  -> ChatResult   [EXISTING substrate]
        (or Agent.run for the tool loop)
    persist Run               -> Repository (runs, snapshots)        [the new trace sink]
Evaluator.score(run)          -> Score                              [manual/metric v0]
API serves runs/experiments   -> dashboard (read models)
re-run: clone spec, swap one ref OR one param, run again            [batch 'control' = M4]
```

**Snapshot vs spec.** Store the **spec** as reproducible truth (refs + params + policy).
*Additionally* store a **resolved snapshot** on the Run — the concrete `system` + rendered
`messages` + the version numbers used — purely for fast display/audit. Spec is truth; snapshot
is a cache. Disagreement = a bug, spec wins.

**Module boundaries** (the dependency arrow points one way; keep it acyclic):
- `agentAPI/` **[EXISTS]** — the substrate: `backend.py` (Message union, ChatResult, Backend
  Protocol, errors), `agent.py` (I/O-free loop), `tools.py` (REGISTRY), the three backends.
- `domain/` **[NEW]** — `Segment`, `SegmentVersion`, `AssemblySpec`, `GenerationParams`,
  `Run`, `Experiment`, `Score`. Pure; no I/O. *Reuses the substrate's `Message`/`ChatResult`
  rather than redefining them.*
- `ports/` **[NEW]** — `Repository`, `Assembler`, `Evaluator` ABCs. (`Backend` already lives in
  `agentAPI`.)
- `adapters/repo/` **[NEW]** — `sqlite3` Repository + row↔model mapping.
- `assembly/` **[NEW]** — the `concat-v1` Assembler producing `(system, messages, budget)`.
- `experiment/` **[NEW]** — `run_assembly`, `run_experiment`, `compare`; drives `Agent.run`.
- `projections/` **[NEW]** — read models + `ProjectionStore` (§7).
- `api/` **[NEW]** — hand-rolled JSON HTTP over projections + write services.
- `eval/` **[NEW]** — `ManualEvaluator` + trivial metric evaluators.

---

## 7. Read side — read models, API, dashboard **[NEW]**

The write side (segments → versions → specs → runs) is normalized + append-only for truth and
reproducibility — a poor shape to *render*. Read models are query-shaped projections built
*from* the truth tables; the truth stays authoritative, projections are rebuildable caches.

### 7.1 Projection seam + planned migration

**Rebuild-on-read now** (pure function of truth tables, trivially correct), behind a
`ProjectionStore` interface so the swap to **update-on-write** is local. The swap's trigger is
**UI-driven control/HITL**: once the UI both initiates a run and needs to show it immediately,
you already hold the new Run at write time, so updating the projection then is natural and kills
read latency. Same insulation discipline as the model/repo ports.

```python
class ProjectionStore(ABC):
    """Read-side access. v1 impl rebuilds-on-read from the Repository; later impl maintains
    materialized projections updated on write (driven by UI control). Returns READ MODELS
    (denormalized), never write-side types. If a materialized view and a rebuild disagree,
    the rebuild wins."""
    @abstractmethod
    def variant_grid(self, experiment_id: str) -> "ExperimentGrid": ...
    @abstractmethod
    def run_detail(self, run_id: str) -> "RunDetailView": ...
    @abstractmethod
    def segment_history(self, segment_id: str) -> "SegmentHistoryView": ...
    @abstractmethod
    def experiment_index(self) -> list["ExperimentSummary"]: ...
```

### 7.2 Read models — the balanced grid (budget + quality + composition, equal weight)

The central view is **one denormalized row per variant** carrying all three axes pre-joined —
not three views stitched in the frontend. The composition **diff** is relational, so it's a
sibling model.

```python
class VariantRow(BaseModel):
    """One grid row: one variant, all three axes pre-joined, aggregated over N trials."""
    assembly_id: str
    label: str                       # the varied thing's identity ("persona v3" / "temp=0.9")
    # budget
    mean_total_tokens: float
    per_type_tokens: dict[str, float]   # SegmentType -> mean tokens (incl. system)
    truncated_any: bool
    # quality
    mean_score: float | None
    score_kind: str | None
    n_trials: int
    # composition / config
    segment_refs: list[SegmentRef]
    params: GenerationParams         # so a params-varied experiment shows the knob per row


class CompositionDiff(BaseModel):
    """How variants differ. Renders as the 'what changed' affordance. Generic enough for
    multi-thing variation later, though v1 varies exactly one."""
    held_fixed: list[SegmentRef]
    varied_dimension: str            # mirrors Experiment.varied_dimension
    per_variant: dict[str, str]      # assembly_id -> human description of its varied value


class ExperimentGrid(BaseModel):
    experiment_id: str
    name: str
    rows: list[VariantRow]
    diff: CompositionDiff


class RunDetailView(BaseModel):
    """Drill-down: the resolved window (system + messages) with per-segment budget, the
    completion, usage, stop_reason, latency, score. Built from the run snapshot so it renders
    without re-resolving versions."""
    run_id: str
    assembly_id: str
    system_text: str | None
    rendered_messages: list[tuple[str, str, int]]  # (role, text, tokens) per message/segment
    completion: str
    tokens_in: int
    tokens_out: int
    stop_reason: str
    latency_ms: int | None
    score: Score | None
    created_at: datetime


class SegmentHistoryView(BaseModel):
    """Versioning surface: a segment's version timeline + where each version was used."""
    segment_id: str
    name: str
    type: SegmentType
    versions: list[tuple[int, datetime, int | None]]   # (version_no, created_at, token_estimate)
    usage_by_version: dict[int, list[str]]             # version_no -> run_ids


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    varied_dimension: str
    n_variants: int
    n_runs: int
    best_variant_label: str | None   # by mean_score if scored
    created_at: datetime
```

### 7.3 API contract (JSON over HTTP)

Read-first in phase 1; write/control endpoints arrive M4–M5. Hand-rolled over `ProjectionStore`
+ write services; no framework beyond a minimal HTTP lib.

**Read (M3):**
- `GET /experiments` → `list[ExperimentSummary]`
- `GET /experiments/{id}/grid` → `ExperimentGrid`
- `GET /runs/{id}` → `RunDetailView`
- `GET /segments` → segment list
- `GET /segments/{id}/history` → `SegmentHistoryView`

**Write — authoring & calibration (M4):**
- `POST /segments` → create
- `POST /segments/{id}/versions` → append version
- `POST /assemblies` → save spec (incl. params)
- `POST /assemblies/{id}/rerun-with-swap` → clone spec, swap one ref **or one param**, run.
  Body: `{ "swap": {"kind": "segment"|"param", ...} }`. Returns the new `Run`.

**Experiment & scoring (M5):**
- `POST /experiments` → create
- `POST /experiments/{id}/run` → `run_experiment`; body `{ "trials_per_variant": N }`
- `POST /runs/{id}/score` → attach a Score

> Write endpoints return the created `Run`(s) directly — when control/HITL lands they're the
> surface that *also* updates projections (§7.1), so the UI gets what it just made without a
> re-fetch.

### 7.4 Dashboard, in prose (the bought/generated frontend's spec)

Four screens, all driven by the endpoints above:
1. **Experiments browser** (`GET /experiments`) — cards: variant/run counts, varied dimension,
   best variant, date. The "previous experiments" storage surface. Click → grid.
2. **Comparison grid** (`GET /experiments/{id}/grid`) — the heart. One row per variant; equal
   visual weight to budget (mean tokens + per-`SegmentType` breakdown, ideally a small stacked
   bar showing where the window fills), quality (mean score, winner highlighted), and
   composition/config (the varied segment version *or* param knob, with `CompositionDiff` as a
   "what changed" affordance).
3. **Run detail** (`GET /runs/{id}`) — the resolved `system` + messages with per-segment token
   attribution, then completion, usage, stop_reason, latency, score. Budget made legible at the
   segment level.
4. **Segment & versioning** (`GET /segments`, `.../history`) — browse by type; per segment, a
   version timeline + "used in these runs / how they scored." Ties a content change to its
   measured effect.

Control (HITL) extends screens 2–4 with **action affordances** over M4–M5 write endpoints —
swap a segment version or param and re-run from the grid; trigger an experiment; score inline.
No new screens; the read models already carry the IDs those actions need.

---

## 8. Cross-cutting decisions (make once, early)

- **Sync core for phase 1.** The substrate is sync; observe + batch re-run is request/response.
  Design `Backend`/experiment so an async variant is a parallel implementation, not a rewrite.
  Pay the async tax when live streaming (a later phase) needs it.
- **DB:** stdlib `sqlite3` behind `Repository`; append-only versions + runs; Postgres later as a
  local swap.
- **IDs:** one scheme everywhere (UUIDv4). For *assembly identity*, a content hash of
  `(ordered_refs, params, policy_id)` enables "run this exact recipe before?" dedup.
- **`meta` typing:** loose dict now (documented per-type convention); tighten later without
  touching segment identity.
- **Token authority:** real counts from `ChatResult` are authoritative for *runs*;
  `token_estimate` is advisory; decide whether the assembler *enforces* a pre-call budget
  (needs a `count_tokens` capability on `Backend`) or merely *reports* post-call (leaner first).
- **`GenerationParams` scope:** the intersection of backend capabilities, never the union —
  the substrate's existing Protocol discipline applied to params.

---

## 9. Milestones (always something working; each builds on the substrate)

1. **M1 — domain + repo + reuse substrate fakes.** `domain/` types (reusing `Message`/
   `ChatResult`), `sqlite3` `Repository`, tests using the substrate's hand-rolled-fake style.
   Proof: author segments, append versions, save/load round-trips, all tested. *Most of the
   abstraction-design learning lives here.*
2. **M2 — assembler + thread `GenerationParams` through `Backend`.** `concat-v1` producing
   `(system, messages, budget)`; add `params` (and the deferred `tools` subset seam) to
   `call_model` across the three existing backends. Proof: `run_assembly(spec)` drives a real
   backend and persists a `Run` with real usage/latency.
3. **M3 — API + observability dashboard (the core).** `ProjectionStore` (rebuild-on-read) +
   JSON API; bought/generated frontend renders the four screens. Proof: *see* what went into a
   window (system + messages), per-segment budget, run history.
4. **M4 — re-run / batch calibration (control v1.5).** Clone a spec, swap one ref *or one
   param*, run again; dashboard shows before/after. Proof: act on what you observed.
5. **M5 — experiment layer + manual evaluator.** `Experiment`, `run_experiment`, `compare`;
   `ManualEvaluator` + one metric. Proof: hold task fixed, vary persona/skill *or* temperature,
   read budget + score side-by-side. *A/B realized.*

**Deferred beyond phase 1** (hooks already in place): variable assembly policy (`policy_id`
stamp), automated relevance/quality scoring (`Evaluator` port), per-run tool/skill subsets
(`tools` seam on `Backend` + `SKILL` segment), live between-inference-step control (new
orchestration over the same `Backend` + `Assembler`), streaming, retries/backoff.

---

## 10. Substrate adaptations checklist (the concrete deltas to existing code)

These are the only *changes* to `pyarchAgent` the merge requires; everything else is additive.

- **[ADAPT] `Backend.call_model`** gains `params: GenerationParams` (the substrate's own
  designed escape hatch) and, when the registry-injection refactor happens, an optional per-call
  `tools` subset. Each backend maps params to its wire form; absent knobs use existing defaults.
- **[ADAPT] (optional) `Backend.count_tokens`** if you choose pre-call budget *enforcement*;
  skip if the assembler only *reports* budget post-call from `ChatResult`.
- **[REUSE] `Message` union & `ChatResult`** become the assembler's output element + the `Run`'s
  usage source — imported by `domain/`, not redefined.
- **[REUSE] `Agent.run` (I/O-free)** is the execution engine the experiment layer drives, exactly
  as `probe_bash.py` already does.
- **[REUSE] error taxonomy, `REGISTRY`, `approve`** unchanged; `REGISTRY` + `approve` are the
  seams skills/tool-subsets and execution policy plug into later.

No existing abstraction is discarded. The merge is mostly *additive*, with two small, already-
anticipated changes to one method signature.

---

## 11. Companion / observation mode (deferred, but designed-around) **[NEW — deferred]**

> **Why this is in the doc.** The project's *primary* goal is unchanged: experiment /
> observe / calibrate (first), then orchestrate / advanced HITL of agent workflows (second),
> for **you and advanced users** who *author* their context. This section records a *second*
> intended deployment — a **companion** for **standard users** who run a normal serving
> session (llama.cpp / ollama UI or endpoint) and want observability over it without changing
> how they work. It is deferred (not in phase-1 scope), but recorded now so the design stays
> migration-free for it. It is a real envisioned work setting, not a hypothetical.

### 11.1 The unifying idea — two sources, one observability core

The two modes split along a single axis: **the source of the `(system, messages, usage)`
tuple** that everything downstream observes.

| | Primary mode (you / advanced) | Companion mode (standard user) |
|---|---|---|
| Source | **originated** — you author + assemble + run | **observed** — capture a third-party session |
| You control inputs? | yes | no |
| Observability (budget, fill, history, diff) | full | **full** |
| Calibration (re-run amended) | yes | only by *lifting* an observed session into an authored one |
| A/B (counterfactual cost/quality) | yes | no — variants were never authored |
| HITL control | yes | no |

The companion experience is a **strict subset** of the advanced dashboard — same screens (§7.4),
same read models (§7.2), fed by a thinner source. This is one observability core with two
ingestion paths, **not** a second product. The standard user can **graduate**: lift an observed
session's window into a real `AssemblySpec` and re-run it through *your* model port, at which
point the A/B + control surfaces light up. The architecture encodes the upgrade path.

### 11.2 The one new seam — an ingestion adapter beside `run_assembly`

Observation adds a new *way to produce a `Run`*, sitting beside `run_assembly` (§5), not inside
it. Everything downstream (projections, the four screens, the API) is unchanged.

```
[capture]  llama.cpp / ollama HTTP traffic   (the user points their client at your proxy)
   -> [parse]    vendor wire form -> neutral (system, messages, ChatResult-ish)
   -> [segment]  trivial v0: system -> 1 PERSONA segment; each message -> 1 WORKING_MEMORY
   -> [persist]  Run(source="observed", assembly_id=None)
   -> existing projections + dashboard, UNCHANGED
```

The **parse** step is the *inbound inverse* of the adapters you already wrote: backends do
neutral→wire (`_to_<vendor>_messages`); observation does wire→neutral. For ollama/llama.cpp you
already know that wire shape, so the anti-corruption layer is half-built for this direction.
This is the payoff of the neutral domain model being genuinely neutral, not Anthropic-shaped.

### 11.3 Migration-free hooks to add NOW (cheap insurance, like `policy_id`)

- **`Run.source: str`** — `"generated" | "observed"`. One field; lets projections and the
  dashboard distinguish (and lets the companion UI hide the A/B affordances honestly).
- **`Run.assembly_id: str | None`** — nullable, because an observed run has no `AssemblySpec`
  you authored. (Generated runs always have one.)

Both are additive fields decided now so the companion mode is a feature addition, never a
schema migration. Nothing else in §1–§10 changes.

### 11.4 The two parts to keep contained (same discipline as the rest)

- **Capture = build a proxy, NOT a sniffer.** Sit between the user's UI and their server as a
  *declared hop* (a pass-through the user points their endpoint at). You see plaintext because
  you're in the path — capture is "log what passes through," not packet/TLS interception. This
  stays within the dependency discipline (a pass-through, not a capture library), is portable,
  and is honest software rather than a hack. The cost is a small, upfront UX ask: the user
  re-points their endpoint at you.
- **Observed-segmentation = trivial-and-honest v0.** When *you* assemble, segment identity is
  ground truth (you put the tokens there). When you *observe* a monolithic window, recovering
  "persona vs. retrieved doc vs. history" is *inference*, not parsing — the same
  relevance/structure-recovery research problem you've contained elsewhere. So v0 is dumb on
  purpose: system→one PERSONA segment, each message→one WORKING_MEMORY segment. You still get
  real budget, per-message attribution, history evolution, and cross-session persona diffing —
  the actual companion value. Smart decomposition of a monolithic prompt into typed
  sub-segments is a LATER pluggable layer behind the same kind of seam as the `Evaluator`.
  **Do not let companion mode smuggle the research problem back in through the side door.**

### 11.5 What transfers, and what degrades

- **Transfers in full:** the observation backend, all read models, the four dashboard screens,
  the projection seam, per-message/per-segment budget, history evolution, cross-session diffing
  of persona / context-management policy. Token usage is *real* (the server reports it).
- **Degrades / requires graduation:** counterfactual A/B ("what would persona v4 cost") is
  impossible in pure observation — you didn't author the variants and can't re-run someone
  else's session deterministically. To A/B an observed session you *graduate* it: lift its
  window into an authored `AssemblySpec` and re-run through your own backend (which is your
  code running it, not llama.cpp). This is a genuinely good feature and a clean loop, but it is
  primary-mode machinery, not observation.

> **Net:** companion mode is a deferred, well-supported extension — a provider-agnostic session
> companion for standard users, fed by an ingestion adapter (proxy-capture + wire→neutral parse)
> producing `source="observed"` runs into the existing dashboard with zero downstream change.
> Two fields now (`source`, nullable `assembly_id`) keep it migration-free. Observability and
> calibration transfer; A/B requires graduating an observed session into an authored one.
