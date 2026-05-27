use schemars::JsonSchema;
use serde::Deserialize;

/// Configuration for the built-in OpenAI provider.
///
/// Endpoint, credentials, and headers are supplied through
/// [`crate::ProviderCommonConfig`]. The official OpenAI provider uses sigma's
/// built-in OpenAI-compatible request mapping without exposing configurable
/// `chat_params`; use `kind = "openai-compatible"` when targeting a compatible
/// provider that needs request-field allow/drop/rename rules.
#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(super) struct OpenAiConfig {}
