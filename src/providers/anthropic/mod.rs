use std::sync::Arc;

use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderName, Method};
use serde_json::{Map, Value, json};

use crate::config::{ChatParameterMap, SecretString};
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::providers::common::{
    header_map_from_config, non_empty_env, parse_response_json,
    signing_header_value as header_value,
};
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
mod state;
mod stream;
mod tools;

use config::{AnthropicConfig, resolve_api_base, resolve_api_key};
use error::anthropic_error_response;
use request::{
    add_beta_header, filter_metadata, infer_beta_headers, insert_header_if_missing,
    is_internal_param, map_reasoning_effort, map_response_format, map_stop_sequences,
    map_token_params, map_tool_choice, map_user_metadata, map_web_search_tool,
    merge_header_beta_values, messages_url, provider_options_contain, translate_anthropic_messages,
};
use response::anthropic_response_to_chat;
use state::reverse_tool_map;
use stream::AnthropicSseStream;
use tools::{apply_tool_choice_name_map, prepare_tools};

const ANTHROPIC_KIND: ProviderKindStatic = ProviderKindStatic::new("anthropic");
const ANTHROPIC_DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_DEFAULT_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const RESPONSE_FORMAT_TOOL_NAME: &str = "json_tool_call";
const TOOL_NAME_MAP_STATE_KEY: &str = "anthropic_tool_name_reverse_map";
const OAUTH_TOKEN_PREFIX: &str = "sk-ant-oat";

const SUPPORTED_CHAT_PARAMS: &[&str] = &[
    "anthropic_beta",
    "cache_control",
    "context_management",
    "container",
    "max_completion_tokens",
    "max_tokens",
    "mcp_servers",
    "output_config",
    "output_format",
    "parallel_tool_calls",
    "reasoning_effort",
    "response_format",
    "speed",
    "stop",
    "stream",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "user",
    "web_search",
];

struct AnthropicProvider {
    id: ProviderId,
    kind: ProviderKind,
    chat: AnthropicChatAdapter,
}

impl AnthropicProvider {
    fn from_config(init: ProviderInit<AnthropicConfig>) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let api_base = resolve_api_base(&init);
        let api_key = resolve_api_key(init.common.api_key.clone());
        let auth_token = init
            .config
            .auth_token
            .clone()
            .or_else(|| non_empty_env("ANTHROPIC_AUTH_TOKEN").map(SecretString::from));
        let headers = header_map_from_config(&init.id, init.common.headers)?;

        if api_key.is_none()
            && auth_token.is_none()
            && !headers.contains_key("x-api-key")
            && !headers.contains_key(AUTHORIZATION)
        {
            return Err(SigmaError::ProviderConfig {
                provider: Some(init.id.clone()),
                message:
                    "anthropic provider requires api_key, ANTHROPIC_API_KEY, auth_token, ANTHROPIC_AUTH_TOKEN, x-api-key, or Authorization header"
                        .to_string(),
            });
        }

        Ok(Arc::new(Self {
            id: init.id.clone(),
            kind: init.kind,
            chat: AnthropicChatAdapter {
                provider: init.id,
                api_base,
                api_key,
                auth_token,
                headers,
                anthropic_version: init
                    .config
                    .anthropic_version
                    .unwrap_or_else(|| ANTHROPIC_DEFAULT_VERSION.to_string()),
                default_max_tokens: init.config.default_max_tokens.unwrap_or_else(|| {
                    non_empty_env("DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS")
                        .and_then(|value| value.parse().ok())
                        .unwrap_or(DEFAULT_MAX_TOKENS)
                }),
                beta: init.config.beta,
            },
        }))
    }
}

impl ProviderDriver for AnthropicProvider {
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

struct AnthropicChatAdapter {
    provider: ProviderId,
    api_base: String,
    api_key: Option<SecretString>,
    auth_token: Option<SecretString>,
    headers: HeaderMap,
    anthropic_version: String,
    default_max_tokens: u32,
    beta: Vec<String>,
}

impl ChatCompletionAdapter for AnthropicChatAdapter {
    fn supported_chat_params(&self) -> Vec<&'static str> {
        SUPPORTED_CHAT_PARAMS.to_vec()
    }

    fn map_chat_params(&self, params: ChatParameterMap) -> SigmaResult<ChatParameterMap> {
        Ok(params)
    }

    fn validate_environment(&self) -> SigmaResult<()> {
        Ok(())
    }

    fn endpoint(&self, _request: &ChatAdapterRequest<'_>) -> SigmaResult<ProviderEndpoint> {
        Ok(ProviderEndpoint {
            method: Method::POST,
            url: messages_url(&self.api_base),
        })
    }

    fn transform_request(
        &self,
        request: ChatAdapterRequest<'_>,
        endpoint: ProviderEndpoint,
    ) -> SigmaResult<ProviderRequest> {
        let mut params = request.params;

        let mut beta_values = self.collect_beta_values(&mut params, request.provider_options);
        let tool_maps = prepare_tools(&mut params)?;
        if tool_maps.has_rewrites() {
            apply_tool_choice_name_map(&mut params, &tool_maps.forward);
        }

        map_token_params(&mut params, self.default_max_tokens);
        map_stop_sequences(&mut params);
        map_reasoning_effort(&mut params, request.context.provider_model)?;
        map_user_metadata(&mut params);
        if provider_options_contain(request.provider_options, "output_format") {
            params.remove("response_format");
        } else {
            map_response_format(&mut params, request.context.provider_model)?;
        }
        map_tool_choice(&mut params);
        map_web_search_tool(&mut params);
        infer_beta_headers(&params, &mut beta_values);
        if let Some(provider_options) = request.provider_options {
            infer_beta_headers(provider_options, &mut beta_values);
        }

        let translated =
            translate_anthropic_messages(&self.provider, request.messages, &tool_maps.forward)?;
        let mut body = Map::new();
        body.insert(
            "model".to_string(),
            Value::String(request.context.provider_model.to_string()),
        );
        body.insert("messages".to_string(), Value::Array(translated.messages));
        if !translated.system.is_empty() {
            body.insert("system".to_string(), Value::Array(translated.system));
        }

        filter_metadata(&mut params);
        for (key, value) in params {
            if !is_internal_param(&key) {
                body.insert(key, value);
            }
        }
        if let Some(provider_options) = request.provider_options {
            for (key, value) in provider_options {
                if !is_internal_param(key) {
                    body.insert(key.clone(), value.clone());
                }
            }
        }

        let mut headers = self.headers.clone();
        insert_header_if_missing(
            &self.provider,
            &mut headers,
            CONTENT_TYPE,
            "application/json",
        )?;
        insert_header_if_missing(&self.provider, &mut headers, ACCEPT, "application/json")?;
        insert_header_if_missing(
            &self.provider,
            &mut headers,
            HeaderName::from_static("anthropic-version"),
            &self.anthropic_version,
        )?;
        merge_header_beta_values(&headers, &mut beta_values);
        if !beta_values.is_empty() {
            let value = beta_values.into_iter().collect::<Vec<_>>().join(",");
            headers.insert(
                HeaderName::from_static("anthropic-beta"),
                header_value(&self.provider, "anthropic-beta", &value)?,
            );
        }

        let body = Value::Object(body);

        let provider_state = if tool_maps.has_rewrites() {
            Some(json!({ TOOL_NAME_MAP_STATE_KEY: tool_maps.reverse }))
        } else {
            None
        };

        Ok(ProviderRequest {
            method: endpoint.method,
            url: endpoint.url,
            headers,
            body,
            provider_state,
        })
    }

    fn sign_request(&self, mut request: ProviderRequest) -> SigmaResult<SignedProviderRequest> {
        if !request.headers.contains_key("x-api-key")
            && !request.headers.contains_key(AUTHORIZATION)
        {
            if let Some(api_key) = &self.api_key {
                if api_key.expose_secret().starts_with(OAUTH_TOKEN_PREFIX) {
                    let value = format!("Bearer {}", api_key.expose_secret());
                    request.headers.insert(
                        AUTHORIZATION,
                        header_value(&self.provider, "authorization", &value)?,
                    );
                    insert_header_if_missing(
                        &self.provider,
                        &mut request.headers,
                        HeaderName::from_static("anthropic-dangerous-direct-browser-access"),
                        "true",
                    )?;
                    add_beta_header(&self.provider, &mut request.headers, "oauth-2025-04-20")?;
                } else {
                    request.headers.insert(
                        HeaderName::from_static("x-api-key"),
                        header_value(&self.provider, "x-api-key", api_key.expose_secret())?,
                    );
                }
            } else if let Some(auth_token) = &self.auth_token {
                let value = format!("Bearer {}", auth_token.expose_secret());
                request.headers.insert(
                    AUTHORIZATION,
                    header_value(&self.provider, "authorization", &value)?,
                );
            }
        }

        Ok(request.into())
    }

    fn transform_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaResult<ChatResponse> {
        let body = parse_response_json(&self.provider, response.body.as_ref())?;
        anthropic_response_to_chat(context, body)
    }

    fn transform_error_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaError {
        anthropic_error_response(context, response)
    }

    fn transform_stream(
        &self,
        context: &ChatAdapterContext<'_>,
        stream: ProviderByteStream,
    ) -> SigmaResult<ChatStream> {
        Ok(Box::pin(AnthropicSseStream::new(
            context.provider.to_owned(),
            stream,
            reverse_tool_map(context),
        )))
    }

    fn stream_behavior(&self) -> StreamBehavior {
        StreamBehavior::native(true)
    }
}

submit_provider! {
    kind: ANTHROPIC_KIND,
    constructor: AnthropicProvider::from_config,
    config: AnthropicConfig,
}
