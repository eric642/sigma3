use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::Deserialize;

use crate::config::{ChatParamConfig, ParamPolicy, SecretString};
use crate::providers::common::non_empty_env;
use crate::{ModelName, ProviderInit, SigmaError, SigmaResult};

/// Configuration for providers that expose an OpenAI-compatible chat
/// completions API.
///
/// This config is intentionally small: endpoint, credentials, and static
/// headers continue to live in [`crate::ProviderCommonConfig`], while
/// OpenAI-compatible parameter compatibility rules live here because they only
/// apply to the OpenAI-compatible request shape.
#[derive(Debug, Clone, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct OpenAiCompatibleConfig {
    /// Chat parameter handling rules for this OpenAI-compatible provider.
    ///
    /// These rules are applied after deployment defaults and request
    /// parameters are merged, before sigma maps semantic parameters into the
    /// OpenAI-compatible request body. Use this to allow provider-specific
    /// fields, drop unsupported OpenAI-compatible fields, rename fields for a
    /// target service, or override rules for a provider-native model.
    #[serde(default)]
    #[schemars(with = "ChatParamConfigSchemaView")]
    pub chat_params: ChatParamConfig,
}

#[derive(Default, JsonSchema)]
#[schemars(deny_unknown_fields)]
#[allow(dead_code)]
struct ChatParamConfigSchemaView {
    /// How sigma handles parameters outside the resolved provider support set.
    #[serde(default = "default_param_policy")]
    policy: ParamPolicy,
    /// Complete semantic chat parameter support set. When omitted, the provider adapter default is used.
    #[serde(default)]
    #[schemars(!default)]
    supported: Vec<String>,
    /// Additional parameter names accepted as-is.
    #[serde(default)]
    allow: Vec<String>,
    /// Top-level parameter names or nested paths to remove before sending the provider request.
    #[serde(default)]
    drop: Vec<String>,
    /// Top-level source-to-target field renames applied after unsupported-parameter handling.
    #[serde(default)]
    #[schemars(!default)]
    rename: BTreeMap<String, String>,
    /// Exact provider-native model names mapped to model-specific parameter rules.
    #[serde(default)]
    models: BTreeMap<ModelName, ChatParamModelConfigSchemaView>,
}

#[derive(Default, JsonSchema)]
#[schemars(deny_unknown_fields)]
#[allow(dead_code)]
struct ChatParamModelConfigSchemaView {
    /// Model-specific unsupported-parameter policy.
    #[serde(default)]
    #[schemars(!default)]
    policy: ParamPolicy,
    /// Complete support set for this provider-native model.
    #[serde(default)]
    #[schemars(!default)]
    supported: Vec<String>,
    /// Additional accepted parameter names for this provider-native model.
    #[serde(default)]
    allow: Vec<String>,
    /// Top-level parameter names or nested paths to remove for this model.
    #[serde(default)]
    drop: Vec<String>,
    /// Top-level source-to-target field renames for this model.
    #[serde(default)]
    #[schemars(!default)]
    rename: BTreeMap<String, String>,
}

const fn default_param_policy() -> ParamPolicy {
    ParamPolicy::RejectUnsupported
}

/// Static behavior for one OpenAI-compatible provider kind.
///
/// Provider crates can register their own kind with [`crate::submit_provider!`]
/// and delegate construction to [`crate::OpenAiCompatibleProvider`] with a
/// spec. The spec controls environment fallbacks, whether authentication is
/// required, and response compatibility behavior while keeping provider
/// discovery inventory-driven.
#[derive(Debug, Clone, Copy)]
pub struct OpenAiCompatibleProviderSpec {
    /// Default API root used when neither config nor environment supplies one.
    ///
    /// The chat adapter appends `/chat/completions` unless the configured base
    /// already ends with that path.
    pub default_api_base: Option<&'static str>,
    /// Environment variables checked, in order, for an API base URL.
    pub api_base_env: &'static [&'static str],
    /// Environment variables checked, in order, for an API key.
    pub api_key_env: &'static [&'static str],
    /// Whether the provider must have an API key or configured Authorization
    /// header at build time.
    pub requires_authentication: bool,
    /// Whether null `usage.*_tokens` values should be normalized to `0` before
    /// deserializing an OpenAI-compatible response.
    pub sanitize_null_usage_tokens: bool,
}

pub(super) fn resolve_api_base<TConfig>(
    init: &ProviderInit<TConfig>,
    spec: OpenAiCompatibleProviderSpec,
) -> SigmaResult<String> {
    init.common
        .api_base
        .clone()
        .or_else(|| first_non_empty_env(spec.api_base_env))
        .or_else(|| spec.default_api_base.map(str::to_string))
        .ok_or_else(|| SigmaError::ProviderConfig {
            provider: Some(init.id.clone()),
            message: required_api_base_message(&init.kind, spec.api_base_env),
        })
}

pub(super) fn resolve_api_key(
    api_key: Option<SecretString>,
    spec: OpenAiCompatibleProviderSpec,
) -> Option<SecretString> {
    api_key.or_else(|| first_non_empty_env(spec.api_key_env).map(SecretString::from))
}

fn first_non_empty_env(names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| non_empty_env(name))
}

fn required_api_base_message(kind: &crate::ProviderKind, api_base_env: &[&str]) -> String {
    if api_base_env.is_empty() {
        format!("{kind} provider requires api_base")
    } else {
        format!(
            "{kind} provider requires api_base or one of {}",
            api_base_env.join(", ")
        )
    }
}
