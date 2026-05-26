# sigma

A general-purpose LLM API client.

## Important Notes

1. As long as this line is not removed, the repository is unreleased. Breaking changes are welcome — prioritize clean and elegant code.
2. Use TDD: migrate test cases first, then implementation. Run `make verify` after completing code changes.
3. This is a public SDK. Public APIs must have meaningful rustdoc comments and usage guidance. Do not add bare exported types, traits, methods, macros, or error variants without documenting their purpose, routing/config semantics, errors, and expected extension points.

## Architecture Rules

- Scope is chat-only and async-only unless a design explicitly expands it.
- Preserve the public namespace call shape: `client.chat().create(&request).await` and `client.chat().create_stream(&request).await`.
- Keep the layering explicit: `Client namespace -> Deployment routing -> ProviderDriver -> ChatCompletionAdapter/BaseConfig hooks -> reqwest HTTP execution`.
- Provider discovery is inventory-driven. Do not add manual provider factory registration APIs to `Client` or `ClientBuilder`.
- Provider registrations must be static, linkable data collected by inventory in core and submitted through `submit_provider!`. Use function pointers for constructors, not closures or runtime registries.
- Inventory iteration order is not stable. Building the provider catalog must reject duplicate provider kinds instead of relying on registration order.
- `ClientBuilder` owns only runtime resources such as `with_http_client(...)`; provider instances are created from `ClientConfig` during `build(...)`.
- Model routing is strongly typed. Use `ModelRef::model(...)`, `ModelRef::deployment(...)`, or `ModelRef::provider_model(...)`; do not infer providers from model string prefixes.
- Chat-layer public types live under `src/types/chat`. These structs and enums must be provider-neutral semantic API types, not mirrors of a specific provider's wire schema or names. Provider implementations are responsible for translating chat-layer semantics into their native request/response shapes.
- HTTP execution is provider-independent and owned by `Client` through `reqwest::Client`. Provider-specific behavior belongs in `ProviderDriver`, `ChatCompletionAdapter`, `CustomChatProvider`, or provider configuration.
- The standard chat adapter lifecycle must remain observable and ordered: supported params, message translation, parameter mapping, environment validation, endpoint selection, request transform, signing, HTTP execution, response or stream decoding.
- Use typed `thiserror` errors for configuration, routing, provider transform/signing/response, unsupported parameters, and HTTP failures.
- Prefer focused fake providers in tests over real external providers for core routing, catalog, adapter lifecycle, and streaming behavior.

## Commands

- `make dev` — fast inner loop: `cargo check` + `cargo clippy -D warnings`.
- `make verify` — format, lint, test, and rustdoc-warns-as-errors. Run after code changes.
