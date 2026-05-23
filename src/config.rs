use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DeploymentId, ModelName, ProviderId, ProviderKind};

/// Provider-specific chat parameters after sigma has removed routing fields.
///
/// Adapters receive this map after deployment defaults and request parameters
/// have been merged. Keys are OpenAI-compatible request field names unless the
/// adapter has already transformed them.
pub type ChatParameterMap = serde_json::Map<String, Value>;

/// Redacted secret string for API keys and other provider credentials.
///
/// `Debug` output never prints the secret value. Provider constructors can call
/// [`SecretString::expose_secret`] when they need to build authentication
/// headers or signing material.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SecretString(String);

impl SecretString {
    /// Returns the raw secret value.
    ///
    /// Keep this value out of logs and error messages.
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString(REDACTED)")
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// Policy for OpenAI-compatible request parameters a provider does not support.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamPolicy {
    /// Return [`crate::SigmaError::UnsupportedParams`] before sending a request.
    #[default]
    RejectUnsupported,
    /// Remove unsupported parameters before adapter-specific parameter mapping.
    DropUnsupported,
}

/// Configuration for one initialized provider instance.
///
/// `kind` chooses the registered provider driver. `id` names this configured
/// instance so deployments can route to it. sigma links built-in chat
/// providers for `kind = "openai"` and `kind = "openai-compatible"`; additional
/// provider crates can register their own kinds with [`crate::submit_provider!`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderInstanceConfig {
    /// Stable id for this configured provider instance.
    pub id: ProviderId,
    /// Registered provider kind, for example `"openai"`,
    /// `"openai-compatible"`, or a kind submitted by another provider crate.
    pub kind: ProviderKind,
    /// Optional provider base URL override.
    ///
    /// For `openai`, this defaults to `OPENAI_BASE_URL`, `OPENAI_API_BASE`, or
    /// `https://api.openai.com/v1`. For `openai-compatible`, this must be set
    /// here or through `OPENAI_COMPATIBLE_API_BASE` / `OPENAI_LIKE_API_BASE`.
    /// Base URLs should include any version path, such as `/v1`; sigma appends
    /// `/chat/completions` unless the URL already ends with that endpoint.
    pub api_base: Option<String>,
    /// Optional provider credential.
    ///
    /// For `openai`, sigma falls back to `OPENAI_API_KEY` and requires either a
    /// key or an explicit `Authorization` header. For `openai-compatible`,
    /// sigma falls back to `OPENAI_COMPATIBLE_API_KEY` /
    /// `OPENAI_LIKE_API_KEY`, but local compatible endpoints may omit
    /// credentials entirely.
    pub api_key: Option<SecretString>,
    /// Additional static headers made available to the provider constructor.
    ///
    /// Built-in OpenAI providers send these headers on every request. An
    /// explicit `Authorization` or `Content-Type` header is preserved and is
    /// not overwritten by generated defaults.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Provider-specific configuration that sigma core does not interpret.
    ///
    /// The built-in `openai-compatible` provider accepts
    /// `map_max_completion_tokens_to_max_tokens: bool`, defaulting to `true`,
    /// for endpoints that expect legacy `max_tokens`.
    #[serde(default)]
    pub options: Value,
}

/// Route from a public model name to a provider-native model.
///
/// Deployments are the middle layer between `client.chat()` requests and a
/// provider driver. They let applications expose stable model names while
/// changing provider instances, provider-native model names, defaults, or
/// metadata in configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDeploymentConfig {
    /// Stable deployment id for direct deployment routing.
    pub id: DeploymentId,
    /// Public model name callers use with [`crate::ModelRef::model`].
    pub public_model: ModelName,
    /// Provider instance that handles this deployment.
    pub provider: ProviderId,
    /// Provider-native model name sent to the provider adapter.
    pub provider_model: ModelName,
    /// Default chat parameters merged before request-level parameters.
    #[serde(default)]
    pub defaults: ChatParameterMap,
    /// Opaque model metadata made available to provider adapters.
    #[serde(default)]
    pub model_info: Value,
}

/// Complete runtime configuration for a [`crate::Client`].
///
/// `providers` are instantiated through inventory-registered constructors.
/// `deployments` define routing from public model names to provider instances.
/// `default_model` is used when a request's model is the default empty
/// [`crate::ModelRef`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClientConfig {
    /// Provider instances to initialize during [`crate::Client::build`].
    #[serde(default)]
    pub providers: Vec<ProviderInstanceConfig>,
    /// Deployment routes available to chat requests.
    #[serde(default)]
    pub deployments: Vec<ModelDeploymentConfig>,
    /// Optional public model name used by empty/default request model refs.
    pub default_model: Option<ModelName>,
    /// Behavior for unsupported OpenAI-compatible request parameters.
    #[serde(default)]
    pub param_policy: ParamPolicy,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            providers: Vec::new(),
            deployments: Vec::new(),
            default_model: None,
            param_policy: ParamPolicy::RejectUnsupported,
        }
    }
}
