use std::collections::{BTreeMap, HashMap};
use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{DeploymentId, ModelName, ProviderId, ProviderKind};

/// Provider-specific chat parameters after sigma has removed routing fields.
///
/// Adapters receive this map after deployment defaults and request parameters
/// have been merged. Keys are OpenAI-compatible request field names unless the
/// adapter has already transformed them.
pub type ChatParameterMap = serde_json::Map<String, Value>;

/// Provider-scoped chat request options.
///
/// Keys are configured [`ProviderId`] values, not provider kinds. This keeps
/// protocol drivers such as `"openai-compatible"` separate from the actual
/// provider target, such as `"zhipu"` or `"deepseek"`.
///
/// Values are provider-native JSON object fragments for the selected provider.
/// Standard adapters shallow-merge these fields into the final provider HTTP
/// request body after parameter mapping and after adapter-generated fields such
/// as `"model"`, `"messages"`, or `"stream"`. Provider adapters may also
/// reserve option keys for provider-specific headers or request controls. To
/// send a provider-native OpenAI-style `metadata` field, place it inside the
/// selected provider's options object.
pub type ProviderOptionsMap = HashMap<ProviderId, ChatParameterMap>;

/// Provider-specific configuration as a JSON-compatible object.
///
/// Core sigma does not interpret these fields. Provider registrations bind this
/// map to the typed `config` type passed to [`crate::submit_provider!`] before
/// calling the provider constructor. Because the map uses Serde data model
/// values, callers can populate it from JSON, TOML, YAML, or any other Serde
/// format before building a [`ClientConfig`].
pub type ProviderConfigMap = serde_json::Map<String, Value>;

/// Redacted secret string for API keys and other provider credentials.
///
/// `Debug` output never prints the secret value. Provider constructors can call
/// [`SecretString::expose_secret`] when they need to build authentication
/// headers or signing material.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ParamPolicy {
    /// Return [`crate::SigmaError::UnsupportedParams`] before sending a request.
    #[default]
    RejectUnsupported,
    /// Remove unsupported parameters before adapter-specific parameter mapping.
    DropUnsupported,
}

/// Provider-level chat parameter handling rules.
///
/// These rules are applied by sigma after deployment defaults and request
/// parameters are merged, but before provider request transformation. They let
/// one configured provider instance describe which
/// OpenAI-compatible parameters it accepts, which extra parameters are allowed,
/// which parameters should be removed, and which top-level fields should be
/// renamed before reaching the provider adapter.
///
/// Model-specific entries in [`ChatParamConfig::models`] are matched against
/// the routed provider-native model name, not the public model name. This keeps
/// deployment routing and [`crate::ModelRef::provider_model`] direct routing
/// consistent.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ChatParamConfig {
    /// Policy for parameters not accepted by the resolved support set.
    ///
    /// When this is `None`, sigma defaults to
    /// [`ParamPolicy::RejectUnsupported`]. Model-specific rules may override
    /// this value for one provider-native model.
    #[serde(default)]
    pub policy: Option<ParamPolicy>,
    /// Complete OpenAI-compatible support set for this provider instance.
    ///
    /// When set, this replaces the adapter's built-in support list. Use
    /// [`ChatParamConfig::allow`] to append provider-specific parameters
    /// without replacing the adapter defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supported: Option<Vec<String>>,
    /// Additional parameter names accepted as-is.
    ///
    /// Only parameters actually present in a request or deployment default are
    /// forwarded; entries in this list do not synthesize missing fields.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
    /// Parameter names or nested paths to remove before sending a provider
    /// request.
    ///
    /// Top-level names such as `"logit_bias"` are removed before unsupported
    /// parameter validation. Nested paths such as
    /// `"tools[*].function.parameters.examples"` are removed after field
    /// renaming, using dot notation plus `[*]` or `[0]` array traversal.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop: Vec<String>,
    /// Top-level field renames applied after unsupported parameter handling.
    ///
    /// The map key is the OpenAI-compatible source field and the value is the
    /// provider-native target field. `None` means no configured rename. An empty
    /// map is allowed and explicitly clears inherited rename rules in
    /// model-specific configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rename: Option<BTreeMap<String, String>>,
    /// Provider-native model-specific parameter rules.
    ///
    /// Keys are exact routed provider model names. A matching entry is applied
    /// after the provider-level rules.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<ModelName, ChatParamModelConfig>,
}

/// Provider-native model-specific chat parameter handling rules.
///
/// This has the same behavior as [`ChatParamConfig`] except it cannot contain
/// nested model rules. `supported` and `rename` use `Option` so a model can
/// either inherit provider-level values or replace them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ChatParamModelConfig {
    /// Model-specific unsupported-parameter policy.
    #[serde(default)]
    pub policy: Option<ParamPolicy>,
    /// Complete support set for this provider-native model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supported: Option<Vec<String>>,
    /// Additional accepted parameter names for this model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
    /// Parameter names or nested paths to remove for this model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub drop: Vec<String>,
    /// Top-level field renames for this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rename: Option<BTreeMap<String, String>>,
}

/// Common configuration fields available to every provider instance.
///
/// These fields cover the endpoint, credentials, and static HTTP headers used
/// by most providers. They are serialized at the top level of
/// [`ProviderInstanceConfig`] with Serde `flatten`, so JSON, TOML, and YAML
/// configuration files use `api_base`, `api_key`, `headers`, and
/// `chat_params` beside `id` and `kind`. Provider-specific settings belong in
/// [`ProviderInstanceConfig::config`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
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
    /// Chat parameter handling rules for this provider instance.
    ///
    /// These common rules are interpreted by sigma's standard chat pipeline
    /// before the provider adapter builds the HTTP request. Custom chat
    /// providers that bypass the standard adapter are responsible for applying
    /// any equivalent behavior themselves.
    #[serde(default)]
    pub chat_params: ChatParamConfig,
}

/// Configuration for one initialized provider instance.
///
/// `kind` chooses the registered provider driver. `id` names this configured
/// instance so deployments can route to it. sigma links built-in chat
/// providers for `kind = "openai"` and `kind = "openai-compatible"`; additional
/// provider crates can register their own kinds with [`crate::submit_provider!`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
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
    /// Provider crates should define a typed config struct and register it with
    /// [`crate::submit_provider!`]. sigma uses that type both to deserialize
    /// this object before invoking the constructor and to generate provider
    /// configuration schemas. Common chat parameter support, dropping,
    /// allowing, and renaming rules belong in
    /// [`ProviderCommonConfig::chat_params`].
    #[serde(default)]
    pub config: ProviderConfigMap,
}

/// Route from a public model name to a provider-native model.
///
/// Deployments are the middle layer between `client.chat()` requests and a
/// provider driver. They let applications expose stable model names while
/// changing provider instances, provider-native model names, defaults, or
/// metadata in configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
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
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ClientConfig {
    /// Provider instances to initialize during [`crate::Client::build`].
    #[serde(default)]
    pub providers: Vec<ProviderInstanceConfig>,
    /// Deployment routes available to chat requests.
    #[serde(default)]
    pub deployments: Vec<ModelDeploymentConfig>,
    /// Optional public model name used by empty/default request model refs.
    pub default_model: Option<ModelName>,
}
