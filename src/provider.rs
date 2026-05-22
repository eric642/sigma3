use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures_core::Stream;
use serde_json::Value;

use crate::config::{ChatParameterMap, ProviderInstanceConfig, SecretString};
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
}

inventory::collect!(ProviderRegistration);

/// Catalog of provider constructors discovered from inventory.
///
/// Build code uses this catalog to turn [`crate::ProviderInstanceConfig`] values
/// into initialized [`ProviderDriver`] instances. The order of inventory entries
/// is not stable, so the catalog rejects duplicate provider kinds.
#[derive(Debug, Clone)]
pub struct ProviderCatalog {
    constructors: HashMap<ProviderKind, ProviderConstructor>,
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
        let mut constructors = HashMap::new();

        for registration in registrations {
            let kind = ProviderKind::from(registration.kind);
            if constructors
                .insert(kind.clone(), registration.constructor)
                .is_some()
            {
                return Err(SigmaError::DuplicateProviderRegistration {
                    kind: kind.to_string(),
                });
            }
        }

        Ok(Self { constructors })
    }

    /// Returns whether a provider kind is available in the catalog.
    pub fn contains_kind(&self, kind: &ProviderKind) -> bool {
        self.constructors.contains_key(kind)
    }

    /// Returns the constructor for a provider kind.
    pub fn get(&self, kind: &ProviderKind) -> Option<ProviderConstructor> {
        self.constructors.get(kind).copied()
    }
}

/// Initialization data passed to a provider constructor.
///
/// sigma creates one `ProviderInit` per [`crate::ProviderInstanceConfig`].
/// Provider drivers should validate any provider-specific options here and
/// return [`SigmaError::ProviderConfig`] for invalid configuration.
#[derive(Debug, Clone)]
pub struct ProviderInit {
    /// Configured provider instance id.
    pub id: ProviderId,
    /// Runtime provider kind that matched this constructor.
    pub kind: ProviderKind,
    /// Optional provider base URL override.
    pub api_base: Option<String>,
    /// Optional provider credential.
    pub api_key: Option<SecretString>,
    /// Static headers from configuration.
    pub headers: HashMap<String, String>,
    /// Provider-specific options from configuration.
    pub options: Value,
}

impl From<ProviderInstanceConfig> for ProviderInit {
    fn from(value: ProviderInstanceConfig) -> Self {
        Self {
            id: value.id,
            kind: value.kind,
            api_base: value.api_base,
            api_key: value.api_key,
            headers: value.headers,
            options: value.options,
        }
    }
}

/// Initialized provider instance.
///
/// A driver represents one configured provider instance, including its
/// credentials, base URL, and provider-specific options. Capabilities are
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
#[derive(Debug, Clone)]
pub struct RoutedChatRequest {
    /// Provider instance selected for the request.
    pub provider: ProviderId,
    /// Deployment selected for the request, if routing used one.
    pub deployment: Option<DeploymentId>,
    /// Public model name requested by the caller or deployment.
    pub public_model: ModelName,
    /// Provider-native model name to use.
    pub provider_model: ModelName,
    /// Original chat completion request.
    pub request: CreateChatCompletionRequest,
    /// Opaque model metadata from the selected deployment.
    pub model_info: Value,
}

/// Optional provider capability for fully custom chat handling.
///
/// Implement this when a provider needs to bypass sigma's generic adapter and
/// HTTP execution pipeline. Most HTTP JSON providers should implement
/// [`ChatCompletionAdapter`] instead.
#[async_trait]
pub trait CustomChatProvider: Send + Sync {
    /// Creates one chat completion through provider-specific code.
    async fn create(&self, request: RoutedChatRequest)
    -> SigmaResult<CreateChatCompletionResponse>;

    /// Creates a streaming chat completion through provider-specific code.
    async fn create_stream(&self, request: RoutedChatRequest) -> SigmaResult<ChatStream>;
}

/// Routing metadata shared across chat adapter request and response hooks.
///
/// The context identifies the provider instance, selected deployment, public
/// model name, provider-native model name, and deployment model metadata for a
/// single routed chat request. Adapters receive it again when transforming
/// regular responses or provider byte streams so parsing can depend on the same
/// routing state used to build the request.
#[derive(Debug, Clone)]
pub struct ChatAdapterContext {
    /// Provider instance selected for the request.
    pub provider: ProviderId,
    /// Deployment selected for the request, if routing used one.
    pub deployment: Option<DeploymentId>,
    /// Public model name requested by the caller or deployment.
    pub public_model: ModelName,
    /// Provider-native model name to send to the provider.
    pub provider_model: ModelName,
    /// Opaque model metadata from the selected deployment.
    pub model_info: Value,
}

/// Provider-neutral request data passed through the chat adapter lifecycle.
///
/// The adapter receives translated messages, merged parameters, and routing
/// context. It then chooses an endpoint, serializes a provider request, signs
/// it, and later transforms the response or stream using the same context.
#[derive(Debug, Clone)]
pub struct ChatAdapterRequest {
    /// Routing metadata for this adapter call.
    pub context: ChatAdapterContext,
    /// Provider-specific representation of request messages.
    pub messages: Value,
    /// Merged and policy-filtered chat parameters.
    pub params: ChatParameterMap,
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
    /// sigma uses this list with [`crate::ParamPolicy`] before calling
    /// [`ChatCompletionAdapter::map_openai_params`].
    fn supported_openai_params(&self) -> Vec<&'static str>;

    /// Translates OpenAI-compatible chat messages into provider-specific JSON.
    fn translate_messages(&self, messages: &[ChatCompletionRequestMessage]) -> SigmaResult<Value>;

    /// Maps OpenAI-compatible parameters to provider-specific parameters.
    fn map_openai_params(&self, params: ChatParameterMap) -> SigmaResult<ChatParameterMap>;

    /// Validates credentials or environment needed before each provider call.
    fn validate_environment(&self) -> SigmaResult<()>;

    /// Selects the provider endpoint for a prepared chat request.
    fn endpoint(&self, request: &ChatAdapterRequest) -> SigmaResult<ProviderEndpoint>;

    /// Serializes a prepared chat request into a provider HTTP request.
    fn transform_request(
        &self,
        request: ChatAdapterRequest,
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
        context: &ChatAdapterContext,
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
        context: &ChatAdapterContext,
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
            provider: context.provider.clone(),
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
        context: &ChatAdapterContext,
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

pub(crate) fn deployment_model_info(deployment: Option<&ModelDeploymentConfig>) -> Value {
    deployment
        .map(|deployment| deployment.model_info.clone())
        .unwrap_or(Value::Null)
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
///         // Validate init.options, credentials, and endpoint overrides here.
///         todo!()
///     }
/// }
///
/// submit_provider! {
///     kind: ProviderKindStatic::new("my-provider"),
///     constructor: MyProvider::from_config,
/// }
/// ```
#[macro_export]
macro_rules! submit_provider {
    (kind: $kind:expr, constructor: $constructor:path $(,)?) => {
        $crate::inventory::submit! {
            $crate::ProviderRegistration {
                kind: $kind,
                constructor: $constructor,
            }
        }
    };
}
