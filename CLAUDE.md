# sigma

A general-purpose LLM API client.

## Important Notes

1. As long as this line is not removed, the repository is unreleased. Breaking changes are welcome — prioritize clean and elegant code.
3. Use TDD: migrate test cases first, then implementation. Run `make verify` after completing code changes.

## Commands

- `make dev` — fast inner loop: `cargo check` + `cargo clippy -D warnings`.
- `make verify` — format, lint, test, and rustdoc-warns-as-errors. Run after code changes.
