.PHONY: help init dev verify openai-live anthropic-live ci pre-commit run release clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

init: ## Initialize the development environment (fetch git submodules)
	git submodule update --init --recursive

dev: ## Quick feedback while coding: type-check + clippy
	cargo check --all-targets --all-features
	cargo clippy --all-targets --all-features -- -D warnings

verify: ## After writing code: format, lint, test, and check rustdoc
	cargo fmt --all
	cargo clippy --all-targets --all-features -- -D warnings
	cargo test --all-features
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items

openai-live: ## Run ignored live OpenAI smoke tests (requires OPENAI_API_KEY)
	cargo test --test openai_live -- --ignored --nocapture

anthropic-live: ## Run ignored live Anthropic smoke tests (requires ANTHROPIC_API_KEY)
	cargo test --test anthropic_live -- --ignored --nocapture

pre-commit: ## Before committing: same as verify but fails on unformatted code
	cargo fmt --all -- --check
	cargo clippy --all-targets --all-features -- -D warnings
	cargo test --all-features
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items

ci: pre-commit ## What CI runs

run: ## Run the binary
	cargo run

release: ## Build optimized binary
	cargo build --release

clean: ## Remove build artifacts
	cargo clean
