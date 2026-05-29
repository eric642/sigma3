# sigma — Codebase Optimization Analysis & Roadmap

## Context

`sigma` is an unreleased (~13.3K LOC) general-purpose async LLM API client with a
clean, layered architecture: `Client → deployment routing → ProviderDriver →
ChatCompletionAdapter/BaseConfig hooks → reqwest`. Provider discovery is
inventory-driven; routing is strongly typed via `ModelRef`. Five providers ship:
`anthropic`, `bedrock`, `gemini`, `openai`, `openai-compatible`.

This document is the result of a deep read of the codebase to find optimization
opportunities and a next-step plan. Three parallel exploration passes were run,
then **every high-severity claim was verified directly against source** — which
matters, because several flagged "critical" items turned out to be false
positives. Both the verified findings and the debunked ones are recorded so future
work starts from an accurate baseline.

All code changes are deferred to later, separately-approved steps. This file is
analysis only.

---

## What is already good (do not "fix")

- **Core routing & dispatch** (`src/client.rs`) — correct, readable, well-documented.
- **Inventory catalog** (`src/provider.rs:117-180`) — rejects duplicate provider
  kinds, sorts schemas deterministically; matches CLAUDE.md ordering rules.
- **Adapter lifecycle** — consistently implemented across all five providers; the
  ordered hooks (`endpoint → transform_request → sign_request → transform_response/
  transform_error_response → transform_stream`) are observable and documented at
  `src/provider.rs:500-598`.
- **Typed errors** — `SigmaError` is fully `thiserror`-based with provider
  attribution; the `result_large_err` trade-off is deliberately documented at
  `src/lib.rs:1-6`.
- **SSE chunk-boundary safety** — `SseLineBuffer` (`src/providers/common.rs:123`)
  correctly handles multi-byte UTF-8 split across HTTP chunks.

## False positives from automated exploration (verified — no action)

These were flagged by exploration agents but are **not** real problems:

- **"`panic!`/`unreachable!`/`expect` in production"** — the cited sites in
  `chat_params.rs:304/334` and `openai_compatible/base.rs:225-291/318` are inside
  `#[cfg(test)]` modules. The `unreachable!` at `chat_params.rs:193` and
  `base.rs:101` are guarded invariants (input is provably a JSON object).
- **"`ModelRef` loses provider on round-trip"** — `ModelRef` deserializes to
  `Model` by design so OpenAI-compatible JSON bodies stay unchanged; this is
  documented at `src/model.rs:122-124`. Not a bug.
- **"`read_u32` can panic on short buffer"** (`bedrock/stream.rs:386`) — callers
  guard `buffer.len() < 12` and validate `total_len` before every slice
  (`stream.rs:351-374`). Safe.
- **"Anthropic and Bedrock `request.rs` are ~duplicated, save ~400 lines"** —
  **overstated.** Anthropic speaks the native **Messages** wire format
  (`tool_use`, `input_schema`, `thinking`, `cache_control`, snake_case); Bedrock
  speaks the **Converse** wire format (`toolSpec`, `inferenceConfig`, `toolUse`,
  `reasoningContent`, `cachePoint`, camelCase). The bodies are genuinely
  different. Only *narrow* helpers overlap (see Tier 2).

---

## Findings & Roadmap (prioritized)

### Tier 1 — High value, low risk (recommend first)

#### 1.1 Enforce rustdoc with `#![deny(missing_docs)]`
- **Where:** `src/lib.rs` (only lint today is `allow(clippy::result_large_err)`).
- **Why:** CLAUDE.md mandates rustdoc on every public item, but nothing enforces
  it; gaps have already crept in (see 1.2). A crate-level lint makes the rule
  mechanical and CI-checked, and `make verify` already runs
  `RUSTDOCFLAGS="-D warnings"`.
- **Approach:** add `#![deny(missing_docs)]`, then fix every resulting error
  (bounded, finite list). Keep `#[doc(hidden)]` on the intentionally-internal
  macro-support items (`ProviderInstanceConfigSchemaFn`, `__from_erased`,
  `__private`).

#### 1.2 Fill verified bare public docs
- **`GrammarSyntax`** enum — undocumented (`src/types/shared/custom_grammar_format_param.rs:8`).
  **Keep** `Lark`/`Regex` (generic, self-explanatory); just add enum- and
  variant-level rustdoc.
- **`CompletionTokensDetails::accepted_prediction_tokens`** — bare field
  (`src/types/shared/completion_tokens_details.rs:6`). Keep the name; add a doc
  matching the existing `rejected_prediction_tokens` doc (lines 20-24).
- **`FunctionName`** struct — has a field doc but no struct-level doc
  (`src/types/shared/function_name.rs:3`).
- **`StreamOptions::include_obfuscation`** — has a doc, verify it's meaningful
  (`src/types/chat/options.rs:194`); fine as-is.
- These (plus whatever `#![deny(missing_docs)]` surfaces) are the full set.

#### 1.3 Fix `skip_serializing_if` inconsistency in token-detail structs
- **Where:** `src/types/shared/completion_tokens_details.rs` (fields at lines 6,
  8, 19, 24 lack `skip_serializing_if`; lines 10/13/16 have it) and the parallel
  `prompt_tokens_details.rs`.
- **Why:** inconsistent policy means some `None` fields emit `null` and others are
  omitted — an unstable wire shape for consumers and an arbitrary inconsistency.
- **Approach:** apply `#[serde(skip_serializing_if = "Option::is_none")]`
  uniformly to all optional fields in both structs. Add a serde round-trip test
  asserting absent fields are omitted (no `null`).

#### 1.4 Hoist the shared reasoning-effort → budget table
- **Where:** identical match arms in `anthropic/request.rs:148-154` and
  `bedrock/request.rs:567-579` (`minimal/low→1024, medium→2048, high→4096,
  xhigh→8192, max→16384`).
- **Why:** true duplication of a semantic mapping; drift risk if one provider's
  table is updated.
- **Approach:** add a small `pub(crate)` helper in `src/providers/common.rs`
  (e.g. `reasoning_budget_tokens(effort: &str) -> Option<u32>` returning `None`
  for `"none"`). Each provider keeps its own *wrapping* logic (Anthropic's
  adaptive-mode branch, Bedrock's GptOss/Nova2 branches) but calls the shared
  table for the budget value.

#### 1.5 De-duplicate provider test helpers into `tests/support`
- **Where:** `last_request` / `last_body` are copy-pasted in
  `anthropic_provider.rs:96-106`, `bedrock_provider.rs:164-174`,
  `gemini_provider.rs:92-102`, `openai_provider.rs:132-142`,
  `client_inventory.rs:396`; `response_body` duplicated in `openai_provider.rs:102`
  and `client_inventory.rs:401`.
- **Why:** maintenance burden; bug fixes must be replicated 4-5×.
- **Approach:** create a shared `tests/support/mod.rs` (or `tests/support.rs`)
  with `mock_server`, `last_request`, `last_body`, and a generic `response_body`,
  and `use` it from each integration test (Cargo supports shared modules under
  `tests/` via a non-`.rs`-test submodule). Keep the bespoke `FakeProvider` /
  `FakeChatAdapter` in `client_inventory.rs` — that's the CLAUDE.md-preferred
  focused fake and is correctly scoped.

### Tier 2 — Medium value, modest risk (recommend after Tier 1)

#### 2.1 Collapse the double deployment lookup in `resolve_route`
- **Where:** `src/client.rs:355-394`. The `Deployment` and `Model` arms both end
  in `route_for_deployment`, and the `Model` arm looks up `deployments_by_public_model`
  then `deployments_by_id` with two near-identical `NoDeploymentForModel` error
  sites (lines 361, 382, 390).
- **Approach:** extract `fn deployment_by_id(&self, id, model_for_err) ->
  SigmaResult<ModelDeploymentConfig>` and reuse it from both arms. Net effect:
  one error site, less branching. Low risk; covered by existing routing tests.

#### 2.2 Extract a same-role message-merge helper
- **Where:** `append_anthropic_message` (`anthropic/request.rs:725`) and
  `push_message` (`bedrock/request.rs:161`) are structurally identical (merge
  content into the last message when roles match, else push).
- **Approach:** `pub(crate) fn push_or_merge_role(messages, role, content)` in
  `common.rs`. Note: the *content block shapes* differ per provider, but the
  list-merge mechanic is identical — only that mechanic moves.
- **Caveat:** small win (~15 lines). Bundle with 1.4 since both touch `common.rs`.

#### 2.3 Decompose the largest request builders
- **Where:** `bedrock/request.rs::map_tools` (114 lines, `:619-732`) and
  `map_params` (`:422-500`); `anthropic/request.rs::map_tool_choice` (`:275-329`).
- **Why:** readability/testability only — these are correct but dense.
- **Approach:** split `map_tools` into parse → rewrite → build-config phases (the
  two-pass structure is already there, just inline). Lower priority than 2.1-2.2;
  do only if touching these files for other reasons.

### Tier 3 — Feature gaps & considerations (track, schedule deliberately)

#### 3.1 Bedrock SigV4 credential-chain support (real feature gap)
- **Where:** `src/providers/bedrock/mod.rs:200-202` (`TODO`). Only static
  credentials are supported today — no default provider chain, profiles,
  AssumeRole, web-identity, or refreshable credentials.
- **Why deferred:** CLAUDE.md notes the blocker — `sign_request` is sync, but full
  credential resolution needs async. This is a design task (a provider lifecycle
  extension or an async signing hook), not a quick fix. Worth a dedicated design
  doc before implementation.

#### 3.2 Provider-neutrality — settled
- Policy: field/variant names need not be neutral when sufficiently generic or
  self-explanatory; other providers translate on their end. Under that rule, the
  agent-flagged "neutrality violations" are **not renames**:
  - `GrammarSyntax::{Lark, Regex}`, `ResponseFormat::{Text, JsonObject,
    JsonSchema}`, `CacheControlTtl::{FiveMinutes, OneHour}` — all generic enough;
    **keep**. Their only real defect is missing/uneven rustdoc (covered by 1.1-1.2).
  - `accepted_prediction_tokens` / `rejected_prediction_tokens` — descriptive
    enough; **keep** (fix doc + serde only, per 1.2-1.3).
- **No action beyond Tier 1 documentation.** Recorded here so the question is
  settled and not re-raised.

#### 3.3 Optional: type-validation at construction (defer / discuss)
- Names like `FunctionObject::name`, `ResponseFormatJsonSchema::name` document a
  "≤64 chars, `[a-zA-Z0-9_-]`" contract but don't enforce it; invalid values fail
  only at provider time.
- **Recommendation: do NOT add validating newtypes now.** It's a large public-API
  change for marginal benefit, and providers already sanitize tool names
  (`build_tool_name_rewrites`). Listed only for completeness.

---

## Verification (for future, separately-approved changes)

- **No build/test impact from this document** — it's Markdown.
- **For each future Tier-1/2 change** the standard loop applies: TDD (migrate/add
  the test first), then `make verify` (fmt + clippy `-D warnings` +
  `cargo test --all-features` + rustdoc-as-errors). 1.1's `#![deny(missing_docs)]`
  and `make verify`'s existing `RUSTDOCFLAGS="-D warnings"` together make the
  rustdoc requirement self-enforcing thereafter.

## Recommended execution order (when changes are later approved)

1. 1.1 + 1.2 (docs + lint) — unblocks enforcement, smallest blast radius.
2. 1.3 (serde consistency) — isolated, add round-trip test.
3. 1.4 + 2.2 (shared `common.rs` helpers) — one PR, shared file.
4. 1.5 (`tests/support`) — pure test refactor.
5. 2.1 (`resolve_route`) — isolated routing cleanup.
6. 2.3 (builder decomposition) — opportunistic.
7. 3.1 (Bedrock credential chain) — own design doc first.
