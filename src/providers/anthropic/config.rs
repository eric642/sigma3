use schemars::JsonSchema;
use serde::Deserialize;

use crate::ProviderInit;
use crate::config::SecretString;
use crate::providers::common::non_empty_env;

use super::ANTHROPIC_DEFAULT_BASE_URL;

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(super) struct AnthropicConfig {
    /// Anthropic API version header value.
    pub(super) anthropic_version: Option<String>,
    /// Default `max_tokens` used when the request omits both token limit fields.
    pub(super) default_max_tokens: Option<u32>,
    /// Optional bearer auth token. Prefer `api_key` for normal Anthropic API keys.
    pub(super) auth_token: Option<SecretString>,
    /// Static Anthropic beta header values added to every request.
    pub(super) beta: Vec<String>,
}

pub(super) fn resolve_api_base(init: &ProviderInit<AnthropicConfig>) -> String {
    init.common
        .api_base
        .clone()
        .or_else(|| non_empty_env("ANTHROPIC_API_BASE"))
        .or_else(|| non_empty_env("ANTHROPIC_BASE_URL"))
        .unwrap_or_else(|| ANTHROPIC_DEFAULT_BASE_URL.to_string())
}

pub(super) fn resolve_api_key(api_key: Option<SecretString>) -> Option<SecretString> {
    api_key.or_else(|| non_empty_env("ANTHROPIC_API_KEY").map(SecretString::from))
}
