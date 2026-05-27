use std::sync::Arc;

use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderValue, Method};

use crate::config::{ChatParameterMap, SecretString};
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::providers::common::{header_map_from_config, parse_response_json};
use crate::types::chat::ChatResponse;
use crate::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatStream, ProviderDriver,
    ProviderId, ProviderInit, ProviderKind, ProviderKindStatic, SigmaError, SigmaResult,
    StreamBehavior, submit_provider,
};

mod config;
mod error;
mod request;
mod response;
mod stream;

use config::{
    OpenAiCompatibleConfig, OpenAiConfig, OpenAiFlavor, resolve_api_base, resolve_api_key,
};
use error::openai_error_response;
use request::{chat_completions_url, openai_chat_body, rename_param};
use response::{map_response_reasoning_content, sanitize_null_usage_tokens};
use stream::OpenAiSseStream;

const OPENAI_KIND: ProviderKindStatic = ProviderKindStatic::new("openai");
const OPENAI_COMPATIBLE_KIND: ProviderKindStatic = ProviderKindStatic::new("openai-compatible");
const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

const SUPPORTED_CHAT_PARAMS: &[&str] = &[
    "audio_output",
    "count",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "output_modalities",
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
    "web_search",
];

struct OpenAiProvider {
    id: ProviderId,
    kind: ProviderKind,
    chat: OpenAiChatAdapter,
}

impl OpenAiProvider {
    fn from_openai_config(
        init: ProviderInit<OpenAiConfig>,
    ) -> SigmaResult<Arc<dyn ProviderDriver>> {
        Self::from_config(init, OpenAiFlavor::OpenAi)
    }

    fn from_compatible_config(
        init: ProviderInit<OpenAiCompatibleConfig>,
    ) -> SigmaResult<Arc<dyn ProviderDriver>> {
        Self::from_config(init, OpenAiFlavor::Compatible)
    }

    fn from_config<TConfig>(
        init: ProviderInit<TConfig>,
        flavor: OpenAiFlavor,
    ) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let api_base = resolve_api_base(&init, flavor)?;
        let api_key = resolve_api_key(init.common.api_key.clone(), flavor);
        let headers = header_map_from_config(&init.id, init.common.headers)?;

        if flavor.requires_authentication()
            && api_key.is_none()
            && !headers.contains_key(AUTHORIZATION)
        {
            return Err(SigmaError::ProviderConfig {
                provider: Some(init.id.clone()),
                message:
                    "openai provider requires api_key, OPENAI_API_KEY, or an Authorization header"
                        .to_string(),
            });
        }

        Ok(Arc::new(Self {
            id: init.id.clone(),
            kind: init.kind,
            chat: OpenAiChatAdapter {
                provider: init.id,
                api_base,
                api_key,
                headers,
                flavor,
            },
        }))
    }
}

impl ProviderDriver for OpenAiProvider {
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

struct OpenAiChatAdapter {
    provider: ProviderId,
    api_base: String,
    api_key: Option<SecretString>,
    headers: HeaderMap,
    flavor: OpenAiFlavor,
}

impl ChatCompletionAdapter for OpenAiChatAdapter {
    fn supported_chat_params(&self) -> Vec<&'static str> {
        SUPPORTED_CHAT_PARAMS.to_vec()
    }

    fn map_chat_params(&self, mut params: ChatParameterMap) -> SigmaResult<ChatParameterMap> {
        rename_param(&mut params, "audio_output", "audio");
        rename_param(&mut params, "count", "n");
        rename_param(&mut params, "output_modalities", "modalities");
        rename_param(&mut params, "web_search", "web_search_options");

        Ok(params)
    }

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
        let body = openai_chat_body(
            &self.provider,
            request.context.provider_model,
            request.messages,
            &request.params,
            request.provider_options,
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
        if self.flavor.sanitizes_usage() {
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
            self.flavor,
        )))
    }

    fn stream_behavior(&self) -> StreamBehavior {
        StreamBehavior::native(true)
    }
}

submit_provider! {
    kind: OPENAI_KIND,
    constructor: OpenAiProvider::from_openai_config,
    config: OpenAiConfig,
}

submit_provider! {
    kind: OPENAI_COMPATIBLE_KIND,
    constructor: OpenAiProvider::from_compatible_config,
    config: OpenAiCompatibleConfig,
}
