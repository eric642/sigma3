use std::sync::Arc;

use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderValue, Method};

use crate::config::SecretString;
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::providers::common::{header_map_from_config, parse_response_json};
use crate::types::chat::ChatResponse;
use crate::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatStream, ProviderDriver,
    ProviderId, ProviderInit, ProviderKind, ProviderKindStatic, SigmaError, SigmaResult,
    submit_provider,
};

mod base;
mod config;
mod error;
mod request;
mod response;
mod stream;

use base::{OpenAiBuildContext, OpenAiChatBodyBuilder};
pub use config::{OpenAiCompatibleConfig, OpenAiCompatibleProviderSpec};
use config::{resolve_api_base, resolve_api_key};
use error::openai_error_response;
use request::chat_completions_url;
use response::{map_response_reasoning_content, sanitize_null_usage_tokens};
use stream::OpenAiSseStream;

const OPENAI_COMPATIBLE_KIND: ProviderKindStatic = ProviderKindStatic::new("openai-compatible");
const OPENAI_COMPATIBLE_SPEC: OpenAiCompatibleProviderSpec = OpenAiCompatibleProviderSpec {
    default_api_base: None,
    api_base_env: &["OPENAI_COMPATIBLE_API_BASE", "OPENAI_LIKE_API_BASE"],
    api_key_env: &["OPENAI_COMPATIBLE_API_KEY", "OPENAI_LIKE_API_KEY"],
    requires_authentication: false,
    sanitize_null_usage_tokens: true,
};

const SUPPORTED_CHAT_PARAMS: &[&str] = &[
    "audio",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "n",
    "modalities",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning_effort",
    "response_format",
    "safety_identifier",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "verbosity",
    "web_search_options",
];

/// Reusable provider driver for OpenAI-compatible chat completion endpoints.
///
/// This driver implements sigma's standard async chat adapter lifecycle for
/// services that accept OpenAI-style `/chat/completions` requests and return
/// OpenAI-style response or SSE stream payloads. Provider crates can register a
/// custom provider kind with [`crate::submit_provider!`] and delegate their
/// constructor to [`OpenAiCompatibleProvider::from_init`] with an
/// [`OpenAiCompatibleProviderSpec`].
///
/// ```ignore
/// use std::sync::Arc;
/// use sigma::{
///     OpenAiCompatibleConfig, OpenAiCompatibleProvider, OpenAiCompatibleProviderSpec,
///     ProviderDriver, ProviderInit, ProviderKindStatic, SigmaResult, submit_provider,
/// };
///
/// const ACME_SPEC: OpenAiCompatibleProviderSpec = OpenAiCompatibleProviderSpec {
///     default_api_base: Some("https://api.acme.test/v1"),
///     api_base_env: &["ACME_API_BASE"],
///     api_key_env: &["ACME_API_KEY"],
///     requires_authentication: true,
///     sanitize_null_usage_tokens: true,
/// };
///
/// fn from_config(
///     init: ProviderInit<OpenAiCompatibleConfig>,
/// ) -> SigmaResult<Arc<dyn ProviderDriver>> {
///     OpenAiCompatibleProvider::from_config(init, ACME_SPEC)
/// }
///
/// submit_provider! {
///     kind: ProviderKindStatic::new("acme"),
///     constructor: from_config,
///     config: OpenAiCompatibleConfig,
/// }
/// ```
pub struct OpenAiCompatibleProvider {
    id: ProviderId,
    kind: ProviderKind,
    chat: OpenAiCompatibleChatAdapter,
}

impl OpenAiCompatibleProvider {
    /// Builds an OpenAI-compatible provider from the standard compatible
    /// config.
    ///
    /// # Errors
    ///
    /// Returns [`SigmaError::ProviderConfig`] when the API base cannot be
    /// resolved, when required authentication is missing, or when configured
    /// static headers are invalid.
    pub fn from_config(
        init: ProviderInit<OpenAiCompatibleConfig>,
        spec: OpenAiCompatibleProviderSpec,
    ) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let config = init.config.clone();
        Self::from_init(init, config, spec)
    }

    /// Builds an OpenAI-compatible provider from another provider's typed
    /// config.
    ///
    /// Use this from a provider-specific inventory constructor after extracting
    /// the OpenAI-compatible settings from that provider's config. This keeps
    /// provider discovery static while letting simple providers share sigma's
    /// OpenAI-compatible request, signing, response, and stream handling.
    ///
    /// # Errors
    ///
    /// Returns [`SigmaError::ProviderConfig`] when endpoint or authentication
    /// requirements from `spec` are not satisfied, or when static headers
    /// cannot be parsed.
    pub fn from_init<TConfig>(
        init: ProviderInit<TConfig>,
        config: OpenAiCompatibleConfig,
        spec: OpenAiCompatibleProviderSpec,
    ) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let api_base = resolve_api_base(&init, spec)?;
        let api_key = resolve_api_key(init.common.api_key.clone(), spec);
        let headers = header_map_from_config(&init.id, init.common.headers)?;

        if spec.requires_authentication && api_key.is_none() && !headers.contains_key(AUTHORIZATION)
        {
            return Err(SigmaError::ProviderConfig {
                provider: Some(init.id.clone()),
                message: required_authentication_message(&init.kind, spec.api_key_env),
            });
        }

        Ok(Arc::new(Self {
            id: init.id.clone(),
            kind: init.kind,
            chat: OpenAiCompatibleChatAdapter {
                provider: init.id,
                api_base,
                api_key,
                headers,
                chat_params: config.chat_params,
                sanitize_null_usage_tokens: spec.sanitize_null_usage_tokens,
            },
        }))
    }
}

fn required_authentication_message(kind: &ProviderKind, api_key_env: &[&str]) -> String {
    if api_key_env.is_empty() {
        format!("{kind} provider requires api_key or Authorization header")
    } else {
        format!(
            "{kind} provider requires api_key, Authorization header, or one of {}",
            api_key_env.join(", ")
        )
    }
}

impl ProviderDriver for OpenAiCompatibleProvider {
    fn id(&self) -> &ProviderId {
        &self.id
    }

    fn kind(&self) -> &ProviderKind {
        &self.kind
    }

    fn chat(&self) -> Option<&dyn ChatCompletionAdapter> {
        Some(&self.chat)
    }
}

struct OpenAiCompatibleChatAdapter {
    provider: ProviderId,
    api_base: String,
    api_key: Option<SecretString>,
    headers: HeaderMap,
    chat_params: crate::ChatParamConfig,
    sanitize_null_usage_tokens: bool,
}

impl OpenAiChatBodyBuilder for OpenAiCompatibleChatAdapter {
    fn provider_id(&self) -> &ProviderId {
        &self.provider
    }

    fn default_supported_chat_params(&self) -> &'static [&'static str] {
        SUPPORTED_CHAT_PARAMS
    }
}

impl ChatCompletionAdapter for OpenAiCompatibleChatAdapter {
    fn endpoint(&self, _request: &ChatAdapterRequest<'_>) -> SigmaResult<ProviderEndpoint> {
        Ok(ProviderEndpoint {
            method: Method::POST,
            url: chat_completions_url(&self.api_base),
        })
    }

    fn transform_request(
        &self,
        request: ChatAdapterRequest<'_>,
        endpoint: ProviderEndpoint,
    ) -> SigmaResult<ProviderRequest> {
        let provider_options = request.request.provider_options.get(&self.provider);
        let ctx = OpenAiBuildContext {
            provider: &self.provider,
            provider_model: request.context.provider_model,
            messages: &request.request.messages,
            provider_options,
            streaming: request.streaming,
        };
        let body = self.build_chat_body(
            &ctx,
            request.request,
            request.deployment_defaults,
            Some(&self.chat_params),
        )?;

        let mut headers = self.headers.clone();
        if !headers.contains_key(CONTENT_TYPE) {
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        }

        Ok(ProviderRequest {
            method: endpoint.method,
            url: endpoint.url,
            headers,
            body,
            provider_state: None,
        })
    }

    fn sign_request(&self, mut request: ProviderRequest) -> SigmaResult<SignedProviderRequest> {
        if !request.headers.contains_key(AUTHORIZATION)
            && let Some(api_key) = &self.api_key
        {
            let value = format!("Bearer {}", api_key.expose_secret());
            let value =
                HeaderValue::from_str(&value).map_err(|err| SigmaError::ProviderSigning {
                    provider: self.provider.clone(),
                    message: err.to_string(),
                })?;
            request.headers.insert(AUTHORIZATION, value);
        }

        Ok(request.into())
    }

    fn transform_response(
        &self,
        _context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaResult<ChatResponse> {
        let mut body = parse_response_json(&self.provider, response.body.as_ref())?;
        if self.sanitize_null_usage_tokens {
            sanitize_null_usage_tokens(&mut body);
        }
        map_response_reasoning_content(&mut body);

        serde_json::from_value(body).map_err(|err| SigmaError::ProviderResponse {
            provider: self.provider.clone(),
            message: err.to_string(),
        })
    }

    fn transform_error_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaError {
        openai_error_response(context, response)
    }

    fn transform_stream(
        &self,
        _context: &ChatAdapterContext<'_>,
        stream: ProviderByteStream,
    ) -> SigmaResult<ChatStream> {
        Ok(Box::pin(OpenAiSseStream::new(
            self.provider.clone(),
            stream,
            self.sanitize_null_usage_tokens,
        )))
    }
}

fn from_builtin_config(
    init: ProviderInit<OpenAiCompatibleConfig>,
) -> SigmaResult<Arc<dyn ProviderDriver>> {
    OpenAiCompatibleProvider::from_config(init, OPENAI_COMPATIBLE_SPEC)
}

submit_provider! {
    kind: OPENAI_COMPATIBLE_KIND,
    constructor: from_builtin_config,
    config: OpenAiCompatibleConfig,
}
