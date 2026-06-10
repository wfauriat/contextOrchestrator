# Context Orchestrator — Phase 1 Design Reference (v1)

> Scope lock for this draft: typed **segments** (stable IDs, immutable versions); an
> **assembly** as an ordered set of segment-version references stamped with a constant
> `policy_id`, deterministically reconstructable; four **ports** (model adapter, repository,
> assembler, evaluator); an **experiment layer** that varies *segment content only*; and a
> phase-1 **observe → re-run** data flow.
>
> Deliberately deferred (designed *around*, not *for*): variable assembly policies, live
> between-inference-step control, automated relevance scoring, automated quality judgment.
>
> This document is **interfaces and contracts only**. Signatures and docstrings describe the
> *what*; the *how* (bodies, hard decisions) is yours to author. Where an invariant matters,
> it's called out in prose rather than encoded, so you make the choice consciously.

---

## 0. The one mental model

Everything in the context window is a **budgeted resource competing for the same space**.
Persona, skills/exemplars, knowledge, task spec, and working memory are all just *typed
segments* with different lifetimes and relevance profiles. Your system:

1. **assembles** typed segments into a concrete window (and reports the budget), then
2. **attributes** an outcome + quality back to the exact assembly that produced it.

The **assembly** — not the run — is the unit of analysis. A run is one execution; an assembly
is the specific, reproducible composition of segment versions that produced it. A/B testing
means holding the task fixed, varying one segment, and comparing outcomes against assemblies.

The single biggest constraint flowing from that: **an assembly must be fully reconstructable
from stored references** (segment IDs + versions + `policy_id`), not snapshotted as opaque
text. Reproducibility is the spine.

---

## 1. Segment & version types

A **segment** is a named, typed container that *owns a history of immutable versions*. The
segment is the stable identity; a **segment version** is a frozen snapshot of content +
metadata at a point in time. You edit a segment by appending a new version, never by mutating
an existing one.

### Invariants (prose, enforce as you author)
- **Stable ID**: a segment's `id` never changes across its lifetime.
- **Immutable versions**: once written, a `SegmentVersion`'s content and metadata are frozen.
  "Editing" = append a new version with an incremented `version_no`.
- **Monotonic versioning**: `version_no` strictly increases within a segment.
- **Provenance**: every version records where it came from (hand-authored, derived from
  another version, imported). This is what makes the history auditable later.

```python
from enum import Enum
from datetime import datetime
from pydantic import BaseModel


class SegmentType(str, Enum):
    """The category of a segment. Drives default ordering & budgeting downstream,
    but carries no behavior itself — it's a tag the assembler reads."""
    PERSONA = "persona"          # system-prompt region; most static, highest priority
    SKILL = "skill"              # ICL exemplars / capability injections
    KNOWLEDGE = "knowledge"      # retrieved / reference material (RAG-ish)
    TASK_SPEC = "task_spec"      # the concrete instruction for this run
    WORKING_MEMORY = "working_memory"  # history / scratch / evolving state
    # New types are added here. Adding one must NOT require schema changes elsewhere.


class SegmentVersion(BaseModel):
    """An immutable snapshot of a segment's content + metadata.

    Contract:
      - Frozen after creation. Never mutate; append a new version instead.
      - `token_estimate` is advisory metadata recorded at authoring time; the
        authoritative count comes from the assembler/model port at assembly time.
      - `derived_from` records provenance for auditability (None if hand-authored).
    """
    segment_id: str              # FK to the owning Segment.id
    version_no: int              # strictly increasing within the segment
    content: str                 # the actual text payload
    token_estimate: int | None   # advisory only; not the source of truth
    derived_from: tuple[str, int] | None  # (segment_id, version_no) or None
    created_at: datetime
    # Consider: a free-form `meta: dict` for type-specific fields (e.g. skill tags).
    # Decide whether you want that loose or typed-per-SegmentType. Tradeoff noted below.


class Segment(BaseModel):
    """Stable identity + typed category for a thing that lives in the context window.
    Holds no content directly — content lives in its versions.

    Contract:
      - `id` is stable for life.
      - `latest_version_no` is a denormalized convenience; the repository owns keeping
        it consistent with stored versions (decide: store it, or always derive it?).
    """
    id: str
    type: SegmentType
    name: str                    # human label for the dashboard
    created_at: datetime
    latest_version_no: int | None  # None until first version exists
```

**Decision to make now** (it shapes the schema): is `SegmentVersion.meta` a loose `dict`, or
do you want a typed payload per `SegmentType` (e.g. a `SkillMeta` with tags, a `KnowledgeMeta`
with a source URI)? Loose is faster to start and trivially extensible; typed gives you
validation and better dashboard affordances but more classes up front. Given "engineering
first, stay lean," a loose `meta` now with a *documented* convention per type is the lower-risk
choice — you can tighten to typed later without touching the segment identity model.

---

## 2. The Assembly — first-class, reproducible

An **assembly** is the unit A/B and budget both attach to. It is an *ordered list of
segment-version references* plus the `policy_id` constant. It is NOT the assembled text — the
text is *derived* from the assembly by the assembler, deterministically.

### Invariants
- **Reconstructable**: given the stored `AssemblySpec` + access to the (immutable) referenced
  versions + the named policy, the assembler reproduces the *exact* same window. Determinism
  is mandatory; this is what makes a comparison fair.
- **Reference, don't embed**: store `(segment_id, version_no)` pairs, never the resolved text.
  (You may *cache* resolved text in a snapshot for display — see §5 — but the spec is the
  source of truth.)
- **Stamp the policy**: every assembly records `policy_id` even though only one policy exists
  in v1. One field today; it's the cheap insurance that keeps "vary the policy later" a
  feature addition instead of a migration.

```python
class SegmentRef(BaseModel):
    """A pin to one immutable segment version. The pinning is the point —
    referencing 'latest' would break reproducibility."""
    segment_id: str
    version_no: int


class AssemblySpec(BaseModel):
    """The reproducible recipe for a context window.

    Contract:
      - `ordered_refs` is the exact order the policy will consume (v1 policy = concat
        in this order). Order is significant and part of the identity.
      - `policy_id` names the assembly policy. v1: a constant like "concat-v1".
      - Two AssemblySpecs with identical (ordered_refs, policy_id) MUST assemble to
        identical windows. Hash these two fields to get a stable assembly identity if
        you want dedup / "have I run this before?" lookups.
    """
    id: str
    ordered_refs: list[SegmentRef]
    policy_id: str               # constant in v1; never absent
    created_at: datetime
```

---

## 3. The four ports

Ports are the seams you own. Each is an abstract interface; concrete implementations live
behind them. The rule from your dependency constraints: **vendor SDKs live strictly inside an
adapter, translating to/from *your* domain types at the boundary.** Your core never imports an
SDK type.

### 3.1 Model port — the most important insulation

Define your *own* request/response types. The adapter maps them to whatever the vendor SDK
wants. This is what lets you mock the model in tests, swap providers, or run two providers
side-by-side for comparison without touching the core.

```python
from abc import ABC, abstractmethod


class ModelRequest(BaseModel):
    """Your domain request. Note it takes an already-assembled window, not segments —
    assembly is the assembler's job, not the model port's."""
    window: str                  # the assembled context window (single string in v1)
    # Add knobs you actually use; resist mirroring the whole SDK surface.
    max_tokens: int
    stop: list[str] | None = None


class TokenUsage(BaseModel):
    """Authoritative token accounting for one call. The budget axis is built on this."""
    prompt_tokens: int
    completion_tokens: int
    # total is derivable; store it only if your provider reports it independently.


class ModelResponse(BaseModel):
    completion: str
    usage: TokenUsage
    # Consider: raw latency_ms (cheap, useful as a v0 'quality'/cost signal).
    latency_ms: int | None = None


class ModelPort(ABC):
    """The boundary to any LLM provider. Implementations: a real SDK adapter and a
    deterministic fake for tests. Sync in v1 (see async note in §6)."""

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Authoritative count used by the assembler for budgeting. Keep this on the
        model port because tokenization is provider-specific."""
        ...
```

### 3.2 Repository port — persistence behind an interface

SQLite + stdlib `sqlite3` is plenty for phase 1. Keep it behind this interface so a later
Postgres swap is local. Map rows ↔ pydantic models yourself (no ORM) — that mapping is a good
abstraction exercise, not a dependency.

```python
class Repository(ABC):
    """Owns persistence of segments, versions, assemblies, runs, and snapshots.
    Append-only where the domain is append-only (versions, runs)."""

    # --- segments & versions ---
    @abstractmethod
    def create_segment(self, segment: Segment) -> Segment: ...

    @abstractmethod
    def append_version(self, version: SegmentVersion) -> SegmentVersion:
        """Must reject a version_no that isn't latest+1 for its segment
        (enforces monotonicity). Updates the segment's latest_version_no."""

    @abstractmethod
    def get_version(self, segment_id: str, version_no: int) -> SegmentVersion: ...

    @abstractmethod
    def list_segments(self, type: SegmentType | None = None) -> list[Segment]: ...

    # --- assemblies ---
    @abstractmethod
    def save_assembly(self, spec: AssemblySpec) -> AssemblySpec: ...

    @abstractmethod
    def get_assembly(self, assembly_id: str) -> AssemblySpec: ...

    # --- runs & snapshots (see §4–§5) ---
    @abstractmethod
    def save_run(self, run: "Run") -> "Run": ...

    @abstractmethod
    def get_run(self, run_id: str) -> "Run": ...

    @abstractmethod
    def list_runs(self, assembly_id: str | None = None) -> list["Run"]: ...
```

### 3.3 Assembler port — the heart

Turns an `AssemblySpec` into a concrete window **and** a budget report. v1 behavior is fixed
(concatenate in `ordered_refs` order under `concat-v1`), but the *interface* is honest about
the general shape so a richer policy later is a new implementation, not a new signature.

```python
class BudgetReport(BaseModel):
    """What the assembler reports about the window it built. The budget axis lives here."""
    total_tokens: int
    per_segment: list[tuple[SegmentRef, int]]  # token cost attributed to each segment
    truncated: bool                            # did any segment get cut to fit?
    # Consider: per-SegmentType rollups for the dashboard (derivable from per_segment).


class AssembledWindow(BaseModel):
    text: str                    # the concrete window string fed to ModelRequest.window
    budget: BudgetReport


class Assembler(ABC):
    """Deterministic: same (spec, resolved versions, policy) -> same AssembledWindow.

    v1 implementation contract ('concat-v1'):
      - resolve each SegmentRef to its immutable version via the Repository,
      - concatenate contents in ordered_refs order (with a fixed, documented joiner),
      - count tokens via ModelPort.count_tokens for per-segment + total,
      - apply the fixed truncation rule (decide & document: e.g. none in v1, error if
        over budget — simplest and most honest for a first pass).
    """

    @abstractmethod
    def assemble(self, spec: AssemblySpec) -> AssembledWindow: ...
```

> v1 simplification worth taking: make `concat-v1` **error if over budget** rather than
> silently truncate. Truncation strategy is itself a policy concern you've deferred; erroring
> keeps the assembler honest and surfaces budget problems instead of hiding them.

### 3.4 Evaluator port — the deferred-research seam

Quality/effect is the axis where research risk tries to sneak back. Contain it exactly like
the context-intelligence: a pluggable, **dumb-at-first** port. v0 implementations are a manual
human score and a hardcoded metric. The A/B *machinery* (§4) is pure engineering; the
*judgment* is plugged in here.

```python
class Score(BaseModel):
    """A quality/effect signal attributed to one run's result. Intentionally generic."""
    value: float                 # higher = better, by convention
    kind: str                    # e.g. "human", "exact_match", "contains", "latency_inv"
    note: str | None = None


class Evaluator(ABC):
    """Turns a run's result into a Score. v0: ManualEvaluator (records a human score),
    or a trivial metric (exact match, substring, token cost, latency). The 'smart'
    LLM-judge / relevance scorer is a LATER implementation behind this same interface."""

    @abstractmethod
    def score(self, run: "Run") -> Score: ...
```

---

## 4. The experiment layer — where observability meets A/B

Thin orchestration. Holds a **task fixed**, varies **one segment**, runs **N trials**, records
`(assembly, result, budget, score)` tuples for comparison. No framework — this is a few
functions/classes over the ports above.

```python
class Run(BaseModel):
    """One execution of one assembly. The atomic observability record.

    Contract:
      - References the AssemblySpec by id (reproducible), plus a resolved snapshot for
        display (see §5).
      - Captures usage (authoritative budget) and optionally a Score.
      - Append-only: a Run is never edited after completion.
    """
    id: str
    assembly_id: str
    completion: str
    usage: TokenUsage
    budget: BudgetReport
    score: Score | None          # filled by an Evaluator, possibly after the fact
    created_at: datetime


class Experiment(BaseModel):
    """A fixed task + a set of assemblies that vary in exactly one segment.

    Contract:
      - All member assemblies share the same TASK_SPEC ref (the 'held fixed' part).
      - They differ in exactly one varied SegmentRef (the independent variable).
      - This is the object the dashboard groups runs under for comparison.
    """
    id: str
    name: str
    held_fixed: list[SegmentRef]     # the common backbone (incl. the task spec)
    varied_segment_id: str           # which segment is the independent variable
    assembly_ids: list[str]          # one per variant
    created_at: datetime
```

The orchestration these enable (functions you'll author over the ports):

- `run_assembly(spec) -> Run` — assemble → complete → record. The base unit.
- `run_experiment(experiment, trials_per_variant) -> list[Run]` — loop variants × trials,
  each producing a Run. Pure engineering; the comparison falls out of grouping runs by
  `assembly_id`.
- `compare(experiment) -> <table>` — aggregate runs into per-variant budget + score
  summaries. This is the observability payoff and the A/B readout in one.

---

## 5. Phase-1 data flow & module boundaries

The flow, observe-first:

```
author/edit segments        -> Repository (segments, versions)
build an AssemblySpec        -> Repository (assemblies)        [references, not text]
run_assembly(spec):
    Assembler.assemble(spec) -> AssembledWindow (+ BudgetReport)
    ModelPort.complete(...)  -> ModelResponse (+ TokenUsage)
    persist Run              -> Repository (runs, snapshots)
Evaluator.score(run)         -> Score                          [manual/metric v0]
API serves runs/experiments  -> dashboard (read models)
re-run: clone an AssemblySpec, swap one SegmentRef, run again  [batch 'control' = v1.5]
```

### Snapshot vs spec (resolve the tension cleanly)
Store the **spec** as the reproducible source of truth (references + policy). *Additionally*
store a **resolved snapshot** on the Run — the concrete window text + the version numbers as
they were — purely for fast display and audit. The snapshot is a cache/record; the spec is
truth. If they ever disagree, the spec + immutable versions win, and that disagreement is a bug.

### Suggested module boundaries
- `domain/` — the pydantic types in §1–§4. No I/O, no SDK, no DB. Pure.
- `ports/` — the ABCs in §3 (and `Evaluator`). Pure interfaces.
- `adapters/model/` — vendor SDK adapter + a deterministic `FakeModel` for tests.
- `adapters/repo/` — `sqlite3`-backed `Repository` + row↔model mapping.
- `assembly/` — the `concat-v1` `Assembler`.
- `experiment/` — `run_assembly`, `run_experiment`, `compare`.
- `api/` — the JSON HTTP layer (the contract the bought frontend consumes).
- `eval/` — `ManualEvaluator` + trivial metric evaluators.

The dependency arrow points one way: `adapters`, `assembly`, `experiment`, `api`, `eval`
all depend on `domain` + `ports`; `domain` depends on nothing. Keep that acyclic and the
project stays teachable.

---

## 6. Cross-cutting decisions to make once, early

- **Sync vs async**: lean **sync core for phase 1** (observe + batch re-run is
  request/response). Design `ModelPort` so an async variant is a *parallel implementation*,
  not a rewrite. Pay the async tax only when live streaming (phase 3) actually needs it.
- **DB**: start with stdlib `sqlite3`. Repository interface makes Postgres a local swap later.
  Append-only tables for versions and runs simplify everything (no update paths to reason about).
- **IDs**: pick one scheme (e.g. UUIDv4 strings) and use it everywhere. For *assembly identity*
  specifically, consider a content hash of `(ordered_refs, policy_id)` so "same recipe" is
  detectable.
- **`meta` typing**: loose `dict` now with a documented per-type convention; tighten to typed
  payloads later without touching segment identity.
- **Token authority**: `ModelPort.count_tokens` is the single source of truth for budgeting.
  `SegmentVersion.token_estimate` is advisory only.

---

## 7. Milestones (always something working)

1. **M1 — domain + repo + fake model.** `domain/` types, `sqlite3` `Repository`, `FakeModel`.
   Shippable proof: author segments, append versions, save/load round-trips, all under test.
   No real LLM yet. *This is where most of the abstraction-design learning lives.*
2. **M2 — assembler + real model adapter.** `concat-v1` assembling specs into windows with a
   `BudgetReport`; SDK adapter behind `ModelPort`. Shippable proof: `run_assembly(spec)`
   produces a real `Run` with real `TokenUsage`, persisted.
3. **M3 — API + observability dashboard (the core).** JSON API over runs; bought/generated
   frontend renders segment composition, per-segment budget, and run history. Shippable proof:
   you can *see* what went into a window and what it cost.
4. **M4 — re-run / batch calibration (v1.5 control).** Clone a spec, swap one `SegmentRef`,
   run again; dashboard shows before/after. Shippable proof: act on what you observed.
5. **M5 — experiment layer + manual evaluator.** `Experiment`, `run_experiment`, `compare`;
   `ManualEvaluator` + one trivial metric. Shippable proof: hold task fixed, vary persona/skill,
   read budget + score side-by-side. *This is the A/B feature realized.*

Deferred beyond phase 1 (designed *around*): variable assembly policy (the stamped `policy_id`
is the hook), automated relevance/quality scoring (the `Evaluator` port is the hook), and live
between-inference-step control (a new orchestration over the same `ModelPort` + `Assembler`).

---

## 8. Read side — read models, API contract, dashboard

> Companion to §1–§7. The write side (segments → versions → specs → runs) is optimized for
> **truth and reproducibility**: normalized, append-only. That shape is poor for *rendering*.
> This section adds **read models** (query-shaped projections built from the truth tables) and
> the **JSON API** over them. The frontend is bought/generated; it is described in prose, and
> its only contract is the JSON below.

### 8.1 The projection seam (and the planned migration)

Read models are **derived, rebuildable views** — never a second source of truth. The truth
tables (§1–§4) remain authoritative; a projection can always be discarded and rebuilt from them.

Phase-1 strategy: **rebuild-on-read** — each projection is a pure function of the truth tables,
recomputed per query. No sync logic, trivially correct. But put every projection behind an
interface so the eventual swap to **update-on-write** is local, not a rewrite. The trigger for
that swap is **control/HITL**: once the UI both *initiates* a run and immediately needs to
*show* it, you already hold the new run in hand at write time, so updating the projection then
is natural and kills read latency. Same insulation discipline as the model/repo ports.

```python
from abc import ABC, abstractmethod


class ProjectionStore(ABC):
    """Read-side access. v1 impl computes on read from the Repository. A later impl
    maintains materialized projections, updated as runs/versions are written (driven by
    UI-initiated control). Callers depend only on this interface, never on which it is.

    Contract:
      - Return values are READ MODELS (denormalized, §8.2), never write-side domain types.
      - A projection is always reconstructable from the truth tables; if a materialized
        impl and a rebuild disagree, the rebuild wins (the materialized view has a bug).
    """

    @abstractmethod
    def variant_grid(self, experiment_id: str) -> "ExperimentGrid": ...

    @abstractmethod
    def run_detail(self, run_id: str) -> "RunDetailView": ...

    @abstractmethod
    def segment_history(self, segment_id: str) -> "SegmentHistoryView": ...

    @abstractmethod
    def experiment_index(self) -> list["ExperimentSummary"]: ...
```

### 8.2 Read models (the balanced grid)

The central view weights **budget + quality + composition equally**, so the core read model is
a **single denormalized row per variant** carrying all three axes pre-joined — not three views
stitched in the frontend. The dashboard's hardest query becomes a flat select. The composition
*diff* (what changed across variants) is relational, so it's a sibling model, not a row field.

```python
class VariantRow(BaseModel):
    """One row of the balanced grid: one assembly variant, all three axes pre-joined.
    Aggregates over the N trial runs of this variant."""
    assembly_id: str
    label: str                       # e.g. "persona v3" — the varied segment's identity
    # --- budget axis ---
    mean_total_tokens: float
    per_type_tokens: dict[str, float]   # SegmentType -> mean tokens (rolled up)
    truncated_any: bool
    # --- quality axis ---
    mean_score: float | None         # None until an Evaluator has scored runs
    score_kind: str | None
    n_trials: int
    # --- composition axis (this variant's own makeup; diff is separate) ---
    segment_refs: list[SegmentRef]   # the pinned versions in this variant


class CompositionDiff(BaseModel):
    """Relational view: how variants differ in composition. Renders as the 'what changed'
    column/overlay. In v1 (one varied segment) this is small, but model it generally."""
    held_fixed: list[SegmentRef]               # common backbone across all variants
    varied_segment_id: str
    per_variant: dict[str, SegmentRef]         # assembly_id -> the differing ref


class ExperimentGrid(BaseModel):
    """The central comparison view: balanced grid + the composition diff."""
    experiment_id: str
    name: str
    rows: list[VariantRow]
    diff: CompositionDiff


class RunDetailView(BaseModel):
    """Drill-down for a single run: the resolved window with per-segment budget,
    the completion, usage, and score. Built from the run's snapshot (§5) so it renders
    without re-resolving versions."""
    run_id: str
    assembly_id: str
    window_text: str                 # from the stored snapshot (display cache)
    per_segment: list[tuple[SegmentRef, int, str]]  # (ref, tokens, segment name)
    completion: str
    usage: TokenUsage
    score: Score | None
    created_at: datetime


class SegmentHistoryView(BaseModel):
    """Versioning surface for one segment: its version timeline + where each version
    has been used. Supports 'which version did we run, and how did it do?'."""
    segment_id: str
    name: str
    type: SegmentType
    versions: list[tuple[int, datetime, int | None]]  # (version_no, created_at, token_estimate)
    usage_by_version: dict[int, list[str]]            # version_no -> run_ids that used it


class ExperimentSummary(BaseModel):
    """Index-card view for the experiments list / storage browser."""
    experiment_id: str
    name: str
    n_variants: int
    n_runs: int
    best_variant_label: str | None   # by mean_score if scored, else None
    created_at: datetime
```

### 8.3 API contract (JSON over HTTP)

Thin, read-first in phase 1; the write/control endpoints arrive with M4–M5. Endpoints return
the read models above as JSON. Keep it a hand-rolled layer over `ProjectionStore` + the
write-side services — no framework beyond a minimal HTTP lib if you want one.

**Read (M3):**
- `GET /experiments` → `list[ExperimentSummary]` — the storage browser / past experiments.
- `GET /experiments/{id}/grid` → `ExperimentGrid` — the central balanced-grid comparison.
- `GET /runs/{id}` → `RunDetailView` — drill into one run's window + budget + result.
- `GET /segments` → list of segments (write-side `Segment`, or a thin list view).
- `GET /segments/{id}/history` → `SegmentHistoryView` — the versioning surface.

**Write — authoring & calibration (M4, re-run):**
- `POST /segments` → create a segment (returns `Segment`).
- `POST /segments/{id}/versions` → append a version (returns `SegmentVersion`).
- `POST /assemblies` → save an `AssemblySpec`.
- `POST /assemblies/{id}/rerun-with-swap` → clone spec, swap one `SegmentRef`, run.
  Body: `{ "swap": {"segment_id": ..., "to_version_no": ...} }`. Returns the new `Run`.

**Experiment & scoring (M5):**
- `POST /experiments` → create an `Experiment` (fixed task + varied segment + variants).
- `POST /experiments/{id}/run` → `run_experiment`; body `{ "trials_per_variant": N }`.
- `POST /runs/{id}/score` → attach a manual/metric `Score` to a run.

> When control/HITL lands, the write endpoints are the surface that *also* updates projections
> (the §8.1 migration). That's why `rerun-with-swap` and `experiments/{id}/run` return the
> created `Run`/`Run`s directly — the UI gets what it just made without a re-fetch.

### 8.4 Dashboard, in prose (the frontend's spec)

The bought/generated frontend needs to satisfy four screens, all driven by the endpoints above:

1. **Experiments browser** (`GET /experiments`). A list/grid of past experiments as cards
   showing variant count, run count, best variant, and date — this is the "previous
   experiments" storage ergonomics surface. Click-through to the grid.
2. **Comparison grid** (`GET /experiments/{id}/grid`). The heart. One row per variant; columns
   for budget (mean total tokens + a per-`SegmentType` breakdown, ideally a small stacked bar
   so you *see* where the window fills), quality (mean score, with the winner highlighted), and
   composition (the varied segment's version, with the `CompositionDiff` surfaced as a "what
   changed" affordance). Equal visual weight to the three axes — none buried.
3. **Run detail** (`GET /runs/{id}`). The resolved window rendered with per-segment budget
   attribution (each segment block labeled with its token cost), then the completion, usage,
   and score. This is where budget becomes legible at the segment level.
4. **Segment & versioning** (`GET /segments`, `.../history`). Browse segments by type; per
   segment, a version timeline and "used in these runs / how they scored" — the versioning
   observability that ties a content change to its measured effect.

Control (HITL) extends screens 2–4 with **action affordances** over the M4–M5 write endpoints:
swap a segment version and re-run from the grid; trigger an experiment run; score a run inline.
No new screens — control is buttons on the observability surfaces, which is why the read models
already carry the IDs those actions need.
