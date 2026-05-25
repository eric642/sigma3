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

/// Provider-scoped chat request body overrides.
///
/// Keys are configured [`ProviderId`] values, not provider kinds. This keeps
/// protocol drivers such as `"openai-compatible"` separate from the actual
/// provider target, such as `"zhipu"` or `"deepseek"`.
///
/// Values are shallow top-level JSON object fragments that adapters apply to
/// the final provider HTTP request body after parameter mapping and after
/// adapter-generated fields such as `"model"`, `"messages"`, or `"stream"`.
/// Matching entries have the highest request-body priority. To send a
/// provider-native OpenAI-style `metadata` field, place it inside the selected
/// provider's override object.
pub type ProviderMetadataMap = HashMap<ProviderId, ChatParameterMap>;

/// Provider-specific configuration as a JSON-compatible object.
///
/// Core sigma does not interpret these fields. Provider constructors should
/// call [`crate::ProviderInit::deserialize_config`] to read this map into a
/// provider-owned typed configuration struct. Because the map uses Serde data
/// model values, callers can populate it from JSON, TOML, YAML, or any other
/// Serde format before building a [`ClientConfig`].
pub type ProviderConfigMap = serde_json::Map<String, Value>;

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

/// Common configuration fields available to every provider instance.
///
/// These fields cover the endpoint, credentials, and static HTTP headers used
/// by most providers. They are serialized at the top level of
/// [`ProviderInstanceConfig`] with Serde `flatten`, so JSON, TOML, and YAML
/// configuration files use `api_base`, `api_key`, and `headers` beside `id`
/// and `kind`. Provider-specific settings belong in
/// [`ProviderInstanceConfig::config`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderCommonConfig {
    /// Optional provider base URL override.
    ///
    /// Providers decide whether this value is required and how environment
    /// variable fallbacks are applied. Built-in OpenAI providers use it as the
    /// API root and append `/chat/completions` when needed.
    pub api_base: Option<String>,
    /// Optional provider credential.
    ///
    /// Providers may use this directly, combine it with environment fallback
    /// variables, or ignore it when their protocol uses another authentication
    /// mechanism. Keep this value out of logs by using
    /// [`SecretString::expose_secret`] only when constructing credentials.
    pub api_key: Option<SecretString>,
    /// Additional static headers made available to the provider constructor.
    ///
    /// Providers that use sigma's HTTP adapter usually send these headers on
    /// every request. Header validation is provider-owned because supported
    /// syntax can vary by protocol.
    #[serde(default)]
    pub headers: HashMap<String, String>,
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
    /// Common provider configuration fields serialized at this struct's top
    /// level.
    ///
    /// This keeps hand-written config concise while passing common runtime
    /// settings to providers through [`crate::ProviderInit::common`].
    #[serde(flatten)]
    pub common: ProviderCommonConfig,
    /// Provider-specific configuration that sigma core does not interpret.
    ///
    /// Provider crates should document this object through the
    /// `config_schema` function registered with [`crate::submit_provider!`] and
    /// deserialize it with [`crate::ProviderInit::deserialize_config`]. The
    /// built-in `openai-compatible` provider accepts
    /// `map_max_completion_tokens_to_max_tokens: bool`, defaulting to `true`,
    /// for endpoints that expect legacy `max_tokens`.
    #[serde(default)]
    pub config: ProviderConfigMap,
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
