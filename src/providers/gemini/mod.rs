use std::sync::Arc;

use http::header::CONTENT_TYPE;
use http::{HeaderMap, HeaderName, HeaderValue, Method};
use serde_json::Value;

use crate::config::SecretString;
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::providers::chat_params::{
    apply_chat_param_rules, merge_chat_params, resolve_chat_param_rules,
};
use crate::providers::common::{header_map_from_config, non_empty_env, parse_response_json};
use crate::types::chat::ChatResponse;
use crate::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatStream, ProviderDriver,
    ProviderId, ProviderInit, ProviderKind, ProviderKindStatic, SigmaError, SigmaResult,
    submit_provider,
};

mod config;
mod error;
mod helpers;
mod request;
mod response;
mod stream;

use config::{GeminiApiVersion, GeminiConfig};
use error::gemini_error_response;
use helpers::gemini_model_url;
use request::gemini_request_body;
use response::gemini_response_to_chat_response;
use stream::GeminiSseStream;

const GEMINI_KIND: ProviderKindStatic = ProviderKindStatic::new("gemini");
const GEMINI_DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const GEMINI_API_KEY_HEADER: &str = "x-goog-api-key";

/// Semantic chat parameters this adapter exposes through the Gemini API.
///
/// `parallel_tool_calls` is intentionally absent: Gemini's `generateContent`
/// endpoint does not expose a per-request parallel tool calling toggle, and
/// silently dropping it previously hid configuration mistakes. Callers that
/// still want to forward a provider-native field explicitly can use
/// [`crate::types::chat::ChatRequest::provider_options`].
const SUPPORTED_GEMINI_CHAT_PARAMS: &[&str] = &[
    "audio",
    "frequency_penalty",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "n",
    "modalities",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "service_tier",
    "stop",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "web_search_options",
];

struct GeminiProvider {
    id: ProviderId,
    kind: ProviderKind,
    chat: GeminiChatAdapter,
}

impl GeminiProvider {
    fn from_config(init: ProviderInit<GeminiConfig>) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let api_base = init
            .common
            .api_base
            .clone()
            .or_else(|| non_empty_env("GEMINI_API_BASE"))
            .unwrap_or_else(|| GEMINI_DEFAULT_BASE_URL.to_string());
        let api_key = init
            .common
            .api_key
            .clone()
            .or_else(|| non_empty_env("GOOGLE_API_KEY").map(SecretString::from))
            .or_else(|| non_empty_env("GEMINI_API_KEY").map(SecretString::from));
        let headers = header_map_from_config(&init.id, init.common.headers)?;

        if api_key.is_none() && !headers.contains_key(GEMINI_API_KEY_HEADER) {
            return Err(SigmaError::ProviderConfig {
                provider: Some(init.id.clone()),
                message:
                    "gemini provider requires api_key, GOOGLE_API_KEY, GEMINI_API_KEY, or an x-goog-api-key header"
                        .to_string(),
            });
        }

        Ok(Arc::new(Self {
            id: init.id.clone(),
            kind: init.kind,
            chat: GeminiChatAdapter {
                provider: init.id,
                api_base,
                api_key,
                headers,
                api_version: init.config.api_version,
            },
        }))
    }
}

impl ProviderDriver for GeminiProvider {
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

struct GeminiChatAdapter {
    provider: ProviderId,
    api_base: String,
    api_key: Option<SecretString>,
    headers: HeaderMap,
    api_version: GeminiApiVersion,
}

impl ChatCompletionAdapter for GeminiChatAdapter {
    fn endpoint(&self, request: &ChatAdapterRequest<'_>) -> SigmaResult<ProviderEndpoint> {
        let stream = request.streaming;
        let endpoint = if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let mut url = gemini_model_url(
            &self.api_base,
            self.api_version.segment(request.context.provider_model),
            request.context.provider_model,
            endpoint,
        );
        if stream {
            url.push_str("?alt=sse");
        }

        Ok(ProviderEndpoint {
            method: Method::POST,
            url,
        })
    }

    fn transform_request(
        &self,
        request: ChatAdapterRequest<'_>,
        endpoint: ProviderEndpoint,
    ) -> SigmaResult<ProviderRequest> {
        let mut params = merge_chat_params(
            request.deployment_defaults,
            request.request,
            request.streaming,
        )?;
        let rules = resolve_chat_param_rules(
            SUPPORTED_GEMINI_CHAT_PARAMS,
            None,
            request.context.provider_model,
        );
        apply_chat_param_rules(&self.provider, &mut params, &rules)?;

        let mut body = gemini_request_body(
            &self.provider,
            request.context,
            &request.request.messages,
            &params,
        )?;
        if let Some(provider_options) = request.request.provider_options.get(&self.provider) {
            for (key, value) in provider_options {
                body.insert(key.clone(), value.clone());
            }
        }

        let mut headers = self.headers.clone();
        if !headers.contains_key(CONTENT_TYPE) {
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        }

        Ok(ProviderRequest {
            method: endpoint.method,
            url: endpoint.url,
            headers,
            body: Value::Object(body),
            provider_state: None,
        })
    }

    fn sign_request(&self, mut request: ProviderRequest) -> SigmaResult<SignedProviderRequest> {
        if !request.headers.contains_key(GEMINI_API_KEY_HEADER)
            && let Some(api_key) = &self.api_key
        {
            let value = HeaderValue::from_str(api_key.expose_secret()).map_err(|err| {
                SigmaError::ProviderSigning {
                    provider: self.provider.clone(),
                    message: err.to_string(),
                }
            })?;
            request
                .headers
                .insert(HeaderName::from_static(GEMINI_API_KEY_HEADER), value);
        }

        Ok(request.into())
    }

    fn transform_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaResult<ChatResponse> {
        let body = parse_response_json(&self.provider, response.body.as_ref())?;
        gemini_response_to_chat_response(context, response.headers, body)
    }

    fn transform_error_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaError {
        gemini_error_response(context, response)
    }

    fn transform_stream(
        &self,
        context: &ChatAdapterContext<'_>,
        stream: ProviderByteStream,
    ) -> SigmaResult<ChatStream> {
        Ok(Box::pin(GeminiSseStream::new(
            self.provider.clone(),
            context.provider_model.clone(),
            stream,
        )))
    }
}

submit_provider! {
    kind: GEMINI_KIND,
    constructor: GeminiProvider::from_config,
    config: GeminiConfig,
}
