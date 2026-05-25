use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_core::Stream;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::config::{
    ChatParameterMap, ProviderCommonConfig, ProviderConfigMap, ProviderInstanceConfig,
};
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::types::chat::{
    ChatChoiceStream, ChatCompletionRequestMessage, ChatCompletionStreamResponseDelta,
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use crate::{
    DeploymentId, ModelDeploymentConfig, ModelName, ProviderId, ProviderKind, ProviderKindStatic,
    SigmaError, SigmaResult,
};

/// Function pointer used to initialize one configured provider instance.
///
/// Constructors are plain function pointers so provider registrations can be
/// submitted as static inventory data.
pub type ProviderConstructor = fn(ProviderInit) -> SigmaResult<Arc<dyn ProviderDriver>>;

/// Function pointer used to return a provider-specific configuration schema.
///
/// The returned value should be a JSON Schema object describing the nested
/// [`ProviderInstanceConfig::config`] object for one provider kind. Function
/// pointers keep provider registrations static and inventory-friendly while
/// letting provider crates hand-write schemas or generate them with their own
/// tooling.
pub type ProviderConfigSchemaFn = fn() -> Value;

/// Static provider registration collected by the inventory registry.
///
/// Provider crates normally create registrations with [`crate::submit_provider!`]
/// instead of constructing this type directly.
#[derive(Debug, Clone, Copy)]
pub struct ProviderRegistration {
    /// Provider kind matched against [`crate::ProviderInstanceConfig::kind`].
    pub kind: ProviderKindStatic,
    /// Constructor called for each configured provider instance of this kind.
    pub constructor: ProviderConstructor,
    /// Schema for this provider kind's nested configuration object.
    pub config_schema: ProviderConfigSchemaFn,
}

inventory::collect!(ProviderRegistration);

/// Provider instance configuration schema exposed by a provider catalog.
///
/// Each item combines a registered provider kind with the full JSON Schema for
/// one [`ProviderInstanceConfig`] object of that kind. The schema includes
/// sigma's common provider fields and places provider-owned settings under the
/// nested `config` property. This is intended for documentation generators and
/// config UIs; provider constructors remain the source of truth for runtime
/// validation.
#[derive(Debug, Clone, PartialEq)]
pub struct ProviderInstanceConfigSchema {
    /// Registered provider kind described by this schema.
    pub kind: ProviderKind,
    /// JSON Schema for one provider instance configuration object.
    pub schema: Value,
}

/// Catalog of provider constructors discovered from inventory.
///
/// Build code uses this catalog to turn [`crate::ProviderInstanceConfig`] values
/// into initialized [`ProviderDriver`] instances. The order of inventory entries
/// is not stable, so the catalog rejects duplicate provider kinds.
#[derive(Debug, Clone)]
pub struct ProviderCatalog {
    registrations: HashMap<ProviderKind, ProviderRegistration>,
}

impl ProviderCatalog {
    /// Builds a catalog from all linked inventory registrations.
    ///
    /// # Errors
    ///
    /// Returns [`SigmaError::DuplicateProviderRegistration`] if two linked
    /// crates register the same provider kind.
    pub fn from_inventory() -> SigmaResult<Self> {
        Self::from_registrations(inventory::iter::<ProviderRegistration>.into_iter().copied())
    }

    /// Builds a catalog from explicit registrations.
    ///
    /// This is primarily useful for tests because inventory iteration order is
    /// intentionally unspecified.
    ///
    /// # Errors
    ///
    /// Returns [`SigmaError::DuplicateProviderRegistration`] when more than one
    /// registration uses the same kind.
    pub fn from_registrations(
        registrations: impl IntoIterator<Item = ProviderRegistration>,
    ) -> SigmaResult<Self> {
        let mut catalog = HashMap::new();

        for registration in registrations {
            let kind = ProviderKind::from(registration.kind);
            if catalog.insert(kind.clone(), registration).is_some() {
                return Err(SigmaError::DuplicateProviderRegistration {
                    kind: kind.to_string(),
                });
            }
        }

        Ok(Self {
            registrations: catalog,
        })
    }

    /// Returns whether a provider kind is available in the catalog.
    pub fn contains_kind(&self, kind: &ProviderKind) -> bool {
        self.registrations.contains_key(kind)
    }

    /// Returns the constructor for a provider kind.
    pub fn get(&self, kind: &ProviderKind) -> Option<ProviderConstructor> {
        self.registrations
            .get(kind)
            .map(|registration| registration.constructor)
    }

    /// Returns full provider instance configuration schemas sorted by kind.
    ///
    /// Inventory iteration order is intentionally unspecified, so this method
    /// sorts by [`ProviderKind`] before returning schemas. Each schema describes
    /// one complete [`ProviderInstanceConfig`] shape for a specific kind,
    /// including sigma's common fields and that provider's nested `config`
    /// schema.
    pub fn provider_instance_config_schemas(&self) -> Vec<ProviderInstanceConfigSchema> {
        let mut registrations = self.registrations.iter().collect::<Vec<_>>();
        registrations.sort_by_key(|(kind, _)| *kind);

        registrations
            .into_iter()
            .map(|(kind, registration)| ProviderInstanceConfigSchema {
                kind: kind.clone(),
                schema: provider_instance_config_schema(
                    registration.kind,
                    (registration.config_schema)(),
                ),
            })
            .collect()
    }
}

fn provider_instance_config_schema(kind: ProviderKindStatic, config_schema: Value) -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": format!("{} provider instance", kind.as_str()),
        "type": "object",
        "additionalProperties": false,
        "required": ["id", "kind"],
        "properties": {
            "id": {
                "type": "string",
                "description": "Stable provider instance id used by deployments and direct provider-model routing."
            },
            "kind": {
                "const": kind.as_str(),
                "description": "Registered provider kind used to select the provider constructor."
            },
            "api_base": {
                "type": "string",
                "description": "Optional provider API base URL override."
            },
            "api_key": {
                "type": "string",
                "description": "Optional provider credential."
            },
            "headers": {
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                },
                "default": {},
                "description": "Static headers made available to the provider constructor."
            },
            "chat_params": chat_param_config_schema(),
            "config": config_schema
        }
    })
}

fn chat_param_config_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "default": {},
        "description": "Common chat parameter support, allow, drop, rename, and per-provider-model override rules.",
        "properties": {
            "policy": {
                "type": "string",
                "enum": ["reject_unsupported", "drop_unsupported"],
                "default": "reject_unsupported",
                "description": "How sigma handles parameters outside the resolved provider support set."
            },
            "supported": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Complete OpenAI-compatible parameter support set. When omitted, the provider adapter default is used."
            },
            "allow": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Additional parameter names accepted as-is."
            },
            "drop": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Top-level parameter names or nested paths to remove before sending the provider request."
            },
            "rename": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Top-level source-to-target field renames applied after unsupported-parameter handling."
            },
            "models": {
                "type": "object",
                "additionalProperties": chat_param_model_config_schema(),
                "default": {},
                "description": "Exact provider-native model names mapped to model-specific parameter rules."
            }
        }
    })
}

fn chat_param_model_config_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "default": {},
        "properties": {
            "policy": {
                "type": "string",
                "enum": ["reject_unsupported", "drop_unsupported"],
                "description": "Model-specific unsupported-parameter policy."
            },
            "supported": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Complete support set for this provider-native model."
            },
            "allow": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Additional accepted parameter names for this provider-native model."
            },
            "drop": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
                "description": "Top-level parameter names or nested paths to remove for this model."
            },
            "rename": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Top-level source-to-target field renames for this model."
            }
        }
    })
}

/// Initialization data passed to a provider constructor.
///
/// sigma creates one `ProviderInit` per [`crate::ProviderInstanceConfig`].
/// Provider drivers should validate any provider-specific config here and
/// return [`SigmaError::ProviderConfig`] for invalid configuration.
#[derive(Debug, Clone)]
pub struct ProviderInit {
    /// Configured provider instance id.
    pub id: ProviderId,
    /// Runtime provider kind that matched this constructor.
    pub kind: ProviderKind,
    /// Common provider configuration fields.
    pub common: ProviderCommonConfig,
    /// Provider-specific configuration from the nested `config` object.
    pub config: ProviderConfigMap,
}

impl ProviderInit {
    /// Deserializes the provider-specific `config` object into a typed value.
    ///
    /// Provider constructors should define their own configuration structs,
    /// usually with Serde defaults and `deny_unknown_fields`, then call this
    /// method before initializing runtime state. The error is mapped to
    /// [`SigmaError::ProviderConfig`] with the provider instance id attached.
    ///
    /// # Errors
    ///
    /// Returns [`SigmaError::ProviderConfig`] when the nested `config` object
    /// does not match `T`.
    pub fn deserialize_config<T>(&self) -> SigmaResult<T>
    where
        T: DeserializeOwned,
    {
        serde_json::from_value(Value::Object(self.config.clone())).map_err(|err| {
            SigmaError::ProviderConfig {
                provider: Some(self.id.clone()),
                message: format!("invalid provider config: {err}"),
            }
        })
    }
}

impl From<ProviderInstanceConfig> for ProviderInit {
    fn from(value: ProviderInstanceConfig) -> Self {
        Self {
            id: value.id,
            kind: value.kind,
            common: value.common,
            config: value.config,
        }
    }
}

/// Initialized provider instance.
///
/// A driver represents one configured provider instance, including its
/// credentials, base URL, and provider-specific config. Capabilities are
/// exposed through optional methods such as [`ProviderDriver::chat`].
pub trait ProviderDriver: Send + Sync {
    /// Returns the configured provider instance id.
    fn id(&self) -> &ProviderId;

    /// Returns the provider kind used to create this instance.
    fn kind(&self) -> &ProviderKind;

    /// Returns the standard chat adapter when the provider supports chat.
    fn chat(&self) -> Option<&dyn ChatCompletionAdapter> {
        None
    }

    /// Returns a custom chat handler that bypasses the generic HTTP adapter flow.
    ///
    /// This is intended for providers that cannot be modeled as
    /// messages-to-HTTP-transform-to-response-transform.
    fn custom_chat(&self) -> Option<&dyn CustomChatProvider> {
        None
    }
}

/// Fully routed chat request passed to a [`CustomChatProvider`].
#[derive(Debug, Clone, Copy)]
pub struct RoutedChatRequest<'a> {
    /// Provider instance selected for the request.
    pub provider: &'a ProviderId,
    /// Deployment selected for the request, if routing used one.
    pub deployment: Option<&'a DeploymentId>,
    /// Public model name requested by the caller or deployment.
    pub public_model: &'a ModelName,
    /// Provider-native model name to use.
    pub provider_model: &'a ModelName,
    /// Original chat completion request.
    pub request: &'a CreateChatCompletionRequest,
    /// Opaque model metadata from the selected deployment, when routing used one.
    pub model_info: Option<&'a Value>,
}

/// Optional provider capability for fully custom chat handling.
///
/// Implement this when a provider needs to bypass sigma's generic adapter and
/// HTTP execution pipeline. Most HTTP JSON providers should implement
/// [`ChatCompletionAdapter`] instead. Routed request fields are borrowed for
/// the duration of the call; custom providers should clone only the values they
/// must keep beyond the returned future or stream construction.
#[async_trait]
pub trait CustomChatProvider: Send + Sync {
    /// Creates one chat completion through provider-specific code.
    async fn create(
        &self,
        request: RoutedChatRequest<'_>,
    ) -> SigmaResult<CreateChatCompletionResponse>;

    /// Creates a streaming chat completion through provider-specific code.
    async fn create_stream(&self, request: RoutedChatRequest<'_>) -> SigmaResult<ChatStream>;
}

/// Routing metadata shared across chat adapter request and response hooks.
///
/// The context identifies the provider instance, selected deployment, public
/// model name, provider-native model name, and deployment model metadata for a
/// single routed chat request. Adapters receive it again when transforming
/// regular responses or provider byte streams so parsing can depend on the same
/// routing state used to build the request.
#[derive(Debug, Clone, Copy)]
pub struct ChatAdapterContext<'a> {
    /// Provider instance selected for the request.
    pub provider: &'a ProviderId,
    /// Deployment selected for the request, if routing used one.
    pub deployment: Option<&'a DeploymentId>,
    /// Public model name requested by the caller or deployment.
    pub public_model: &'a ModelName,
    /// Provider-native model name to send to the provider.
    pub provider_model: &'a ModelName,
    /// Opaque model metadata from the selected deployment, when routing used one.
    pub model_info: Option<&'a Value>,
}

/// Provider-neutral request data passed through the chat adapter lifecycle.
///
/// The adapter receives translated messages, merged parameters, and routing
/// context. It then chooses an endpoint, applies provider-scoped body overrides
/// while the body is still structured data, serializes a provider request,
/// signs it, and later transforms the response or stream using the same
/// context.
#[derive(Debug, Clone)]
pub struct ChatAdapterRequest<'a> {
    /// Routing metadata for this adapter call.
    pub context: ChatAdapterContext<'a>,
    /// Provider-specific representation of request messages.
    pub messages: Value,
    /// Merged and policy-filtered chat parameters.
    pub params: ChatParameterMap,
    /// Provider-scoped final request body overrides for the selected provider.
    ///
    /// This is [`None`] when the request has no metadata entry for the selected
    /// provider.
    ///
    /// Adapters should apply these shallow top-level fields after provider
    /// parameter mapping and after adapter-generated fields such as `"model"`,
    /// `"messages"`, or `"stream"`, but before serializing the request body.
    pub body_overrides: Option<&'a ChatParameterMap>,
}

/// Stream of OpenAI-compatible chat completion chunks returned by sigma.
pub type ChatStream =
    Pin<Box<dyn Stream<Item = SigmaResult<CreateChatCompletionStreamResponse>> + Send + 'static>>;

/// Streaming strategy used by a chat adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// Use the HTTP streaming path and let the adapter transform provider bytes.
    Native,
    /// Use a non-streaming provider response and emit one synthesized chunk.
    FakeFromResponse,
}

/// Controls how sigma prepares and executes streaming requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamBehavior {
    /// Stream execution mode.
    pub mode: StreamMode,
    /// Whether sigma should add `"stream": true` before parameter mapping.
    pub inject_stream: bool,
}

impl StreamBehavior {
    /// Uses the HTTP streaming path.
    pub const fn native(inject_stream: bool) -> Self {
        Self {
            mode: StreamMode::Native,
            inject_stream,
        }
    }

    /// Uses `execute` and converts the full response into a one-item stream.
    pub const fn fake_from_response() -> Self {
        Self {
            mode: StreamMode::FakeFromResponse,
            inject_stream: false,
        }
    }
}

impl Default for StreamBehavior {
    fn default() -> Self {
        Self::native(true)
    }
}

/// Provider adapter for the generic chat HTTP pipeline.
///
/// The lifecycle for `create` is:
///
/// 1. [`ChatCompletionAdapter::supported_openai_params`]
/// 2. [`ChatCompletionAdapter::translate_messages`]
/// 3. [`ChatCompletionAdapter::map_openai_params`]
/// 4. [`ChatCompletionAdapter::validate_environment`]
/// 5. [`ChatCompletionAdapter::endpoint`]
/// 6. [`ChatCompletionAdapter::transform_request`]
/// 7. [`ChatCompletionAdapter::sign_request`]
/// 8. sigma sends the signed request with its shared [`reqwest::Client`]
/// 9. non-success HTTP statuses use
///    [`ChatCompletionAdapter::transform_error_response`]
/// 10. success responses use [`ChatCompletionAdapter::transform_response`]
///
/// `create_stream` follows the same preparation path. Native streams use
/// [`ChatCompletionAdapter::transform_error_response`] for non-success HTTP
/// statuses or [`ChatCompletionAdapter::transform_stream`] to convert
/// successful provider bytes into OpenAI-compatible chunks. Fake streams use
/// [`ChatCompletionAdapter::transform_response`] and synthesize one chunk from
/// the full success response.
pub trait ChatCompletionAdapter: Send + Sync {
    /// Returns OpenAI-compatible parameter names this provider accepts.
    ///
    /// sigma combines this list with [`crate::ProviderCommonConfig::chat_params`]
    /// before calling [`ChatCompletionAdapter::map_openai_params`].
    fn supported_openai_params(&self) -> Vec<&'static str>;

    /// Translates OpenAI-compatible chat messages into provider-specific JSON.
    fn translate_messages(&self, messages: &[ChatCompletionRequestMessage]) -> SigmaResult<Value>;

    /// Maps OpenAI-compatible parameters to provider-specific parameters.
    fn map_openai_params(&self, params: ChatParameterMap) -> SigmaResult<ChatParameterMap>;

    /// Validates credentials or environment needed before each provider call.
    fn validate_environment(&self) -> SigmaResult<()>;

    /// Selects the provider endpoint for a prepared chat request.
    fn endpoint(&self, request: &ChatAdapterRequest<'_>) -> SigmaResult<ProviderEndpoint>;

    /// Serializes a prepared chat request into a provider HTTP request.
    fn transform_request(
        &self,
        request: ChatAdapterRequest<'_>,
        endpoint: ProviderEndpoint,
    ) -> SigmaResult<ProviderRequest>;

    /// Signs or authenticates the provider request.
    fn sign_request(&self, request: ProviderRequest) -> SigmaResult<SignedProviderRequest>;

    /// Transforms a completed provider HTTP response into a chat response.
    ///
    /// This hook is synchronous by design. The shared [`reqwest::Client`] owned
    /// by [`crate::Client`] has already awaited and buffered the response body.
    /// Providers that need asynchronous response handling should implement
    /// [`CustomChatProvider`] instead of the generic adapter.
    fn transform_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaResult<CreateChatCompletionResponse>;

    /// Transforms a non-success provider HTTP response into a sigma error.
    ///
    /// sigma calls this hook when the provider returns an HTTP status outside
    /// the 2xx success range. The full response body has been buffered so
    /// adapters can parse provider-native error JSON and return
    /// [`SigmaError::ProviderBusiness`] with stable codes, human-readable
    /// messages, and structured details. Transport failures that do not produce
    /// an HTTP response still return [`SigmaError::Http`].
    fn transform_error_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaError {
        let message = if response.body.is_empty() {
            response
                .status
                .canonical_reason()
                .unwrap_or("provider returned unsuccessful HTTP status")
                .to_string()
        } else {
            String::from_utf8_lossy(&response.body).into_owned()
        };

        SigmaError::ProviderBusiness {
            provider: context.provider.to_owned(),
            status: response.status,
            code: None,
            message,
            details: None,
        }
    }

    /// Transforms provider streaming bytes into chat completion chunks.
    ///
    /// The returned stream may synchronously parse, filter, or reshape each raw
    /// provider byte frame. Network polling remains owned by sigma's HTTP
    /// execution path; this hook only defines provider-specific chunk
    /// translation. Providers that need fully custom asynchronous stream
    /// execution should implement [`CustomChatProvider`].
    fn transform_stream(
        &self,
        context: &ChatAdapterContext<'_>,
        stream: ProviderByteStream,
    ) -> SigmaResult<ChatStream>;

    /// Returns streaming behavior for this adapter.
    fn stream_behavior(&self) -> StreamBehavior {
        StreamBehavior::default()
    }
}

pub(crate) fn response_to_stream_chunk(
    response: CreateChatCompletionResponse,
) -> CreateChatCompletionStreamResponse {
    let choices = response
        .choices
        .into_iter()
        .map(|choice| ChatChoiceStream {
            index: choice.index,
            delta: ChatCompletionStreamResponseDelta {
                content: choice.message.content,
                tool_calls: None,
                role: Some(choice.message.role),
                refusal: choice.message.refusal,
            },
            finish_reason: choice.finish_reason,
            logprobs: choice.logprobs,
        })
        .collect();

    CreateChatCompletionStreamResponse {
        id: response.id,
        choices,
        created: response.created,
        model: response.model,
        service_tier: response.service_tier,
        object: "chat.completion.chunk".to_string(),
        usage: response.usage,
    }
}

pub(crate) fn deployment_model_info(deployment: Option<&ModelDeploymentConfig>) -> Option<&Value> {
    deployment.map(|deployment| &deployment.model_info)
}

/// Registers a provider constructor in sigma's distributed inventory.
///
/// Call this macro once per provider kind from the provider crate or module.
/// The constructor is called once for each matching
/// [`ProviderInstanceConfig`] during [`Client::build`](crate::Client::build).
///
/// ```ignore
/// use std::sync::Arc;
/// use sigma::{
///     ProviderDriver, ProviderInit, ProviderKindStatic, SigmaResult, submit_provider,
/// };
///
/// struct MyProvider;
///
/// impl MyProvider {
///     fn from_config(init: ProviderInit) -> SigmaResult<Arc<dyn ProviderDriver>> {
///         // Validate init.config, credentials, and endpoint overrides here.
///         todo!()
///     }
/// }
///
/// fn my_provider_config_schema() -> serde_json::Value {
///     serde_json::json!({
///         "type": "object",
///         "additionalProperties": false,
///         "properties": {
///             "timeout_ms": {
///                 "type": "integer",
///                 "minimum": 1,
///                 "description": "Provider-specific request timeout in milliseconds."
///             }
///         }
///     })
/// }
///
/// submit_provider! {
///     kind: ProviderKindStatic::new("my-provider"),
///     constructor: MyProvider::from_config,
///     config_schema: my_provider_config_schema,
/// }
/// ```
#[macro_export]
macro_rules! submit_provider {
    (kind: $kind:expr, constructor: $constructor:path, config_schema: $config_schema:path $(,)?) => {
        $crate::inventory::submit! {
            $crate::ProviderRegistration {
                kind: $kind,
                constructor: $constructor,
                config_schema: $config_schema,
            }
        }
    };
}
