use std::collections::{HashMap, VecDeque};
use std::env;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use schemars::JsonSchema;
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{ChatParameterMap, SecretString};
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::types::chat::{
    ChatCompletionRequestMessage, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use crate::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatStream, ModelName,
    ProviderDriver, ProviderId, ProviderInit, ProviderKind, ProviderKindStatic, SigmaError,
    SigmaResult, StreamBehavior, submit_provider,
};

const OPENAI_KIND: ProviderKindStatic = ProviderKindStatic::new("openai");
const OPENAI_COMPATIBLE_KIND: ProviderKindStatic = ProviderKindStatic::new("openai-compatible");
const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

const SUPPORTED_OPENAI_CHAT_PARAMS: &[&str] = &[
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "modalities",
    "n",
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
    "user",
    "verbosity",
    "web_search_options",
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

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
struct OpenAiConfig {}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
struct OpenAiCompatibleConfig {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenAiFlavor {
    OpenAi,
    Compatible,
}

impl OpenAiFlavor {
    fn requires_authentication(self) -> bool {
        self == Self::OpenAi
    }

    fn sanitizes_usage(self) -> bool {
        matches!(self, Self::Compatible)
    }
}

struct OpenAiChatAdapter {
    provider: ProviderId,
    api_base: String,
    api_key: Option<SecretString>,
    headers: HeaderMap,
    flavor: OpenAiFlavor,
}

struct OpenAiChatBody<'a> {
    params: &'a ChatParameterMap,
    provider_model: &'a ModelName,
    messages: &'a [ChatCompletionRequestMessage],
    body_overrides: Option<&'a ChatParameterMap>,
}

impl Serialize for OpenAiChatBody<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut len = self
            .params
            .keys()
            .filter(|key| {
                !is_generated_body_key(key.as_str())
                    && !contains_body_override(self.body_overrides, key.as_str())
            })
            .count();

        if !contains_body_override(self.body_overrides, "model") {
            len += 1;
        }
        if !contains_body_override(self.body_overrides, "messages") {
            len += 1;
        }
        if let Some(body_overrides) = self.body_overrides {
            len += body_overrides.len();
        }

        let mut map = serializer.serialize_map(Some(len))?;
        for (key, value) in self.params {
            if !is_generated_body_key(key.as_str())
                && !contains_body_override(self.body_overrides, key.as_str())
            {
                map.serialize_entry(key, value)?;
            }
        }
        if !contains_body_override(self.body_overrides, "model") {
            map.serialize_entry("model", self.provider_model)?;
        }
        if !contains_body_override(self.body_overrides, "messages") {
            map.serialize_entry("messages", self.messages)?;
        }
        if let Some(body_overrides) = self.body_overrides {
            for (key, value) in body_overrides {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

fn is_generated_body_key(key: &str) -> bool {
    key == "model" || key == "messages"
}

fn contains_body_override(body_overrides: Option<&ChatParameterMap>, key: &str) -> bool {
    body_overrides.is_some_and(|body_overrides| body_overrides.contains_key(key))
}

impl ChatCompletionAdapter for OpenAiChatAdapter {
    fn supported_openai_params(&self) -> Vec<&'static str> {
        SUPPORTED_OPENAI_CHAT_PARAMS.to_vec()
    }

    fn map_openai_params(&self, params: ChatParameterMap) -> SigmaResult<ChatParameterMap> {
        Ok(params)
    }

    fn validate_environment(&self) -> SigmaResult<()> {
        Ok(())
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
        let body = serde_json::to_value(&OpenAiChatBody {
            params: &request.params,
            provider_model: request.context.provider_model,
            messages: request.messages,
            body_overrides: request.body_overrides,
        })
        .map_err(|err| SigmaError::ProviderTransform {
            provider: self.provider.clone(),
            message: err.to_string(),
        })?;

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
    ) -> SigmaResult<CreateChatCompletionResponse> {
        let mut body = parse_response_json(&self.provider, response.body.as_ref())?;
        if self.flavor.sanitizes_usage() {
            sanitize_null_usage_tokens(&mut body);
        }

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

struct OpenAiSseStream {
    provider: ProviderId,
    stream: ProviderByteStream,
    buffer: String,
    pending: VecDeque<SigmaResult<CreateChatCompletionStreamResponse>>,
    done: bool,
    flavor: OpenAiFlavor,
}

impl OpenAiSseStream {
    fn new(provider: ProviderId, stream: ProviderByteStream, flavor: OpenAiFlavor) -> Self {
        Self {
            provider,
            stream,
            buffer: String::new(),
            pending: VecDeque::new(),
            done: false,
            flavor,
        }
    }

    fn push_chunk(&mut self, chunk: Bytes) {
        match std::str::from_utf8(&chunk) {
            Ok(text) => {
                self.buffer.push_str(&text.replace("\r\n", "\n"));
                self.drain_buffer(false);
            }
            Err(err) => {
                self.done = true;
                self.pending.push_back(Err(SigmaError::ProviderResponse {
                    provider: self.provider.clone(),
                    message: err.to_string(),
                }));
            }
        }
    }

    fn drain_buffer(&mut self, flush: bool) {
        while let Some(index) = self.buffer.find("\n\n") {
            let event = self.buffer[..index].to_string();
            self.buffer.drain(..index + 2);
            self.push_event(&event);
            if self.done {
                return;
            }
        }

        self.drain_raw_json_lines();

        if flush {
            let event = self.buffer.trim().to_string();
            self.buffer.clear();
            if !event.is_empty() {
                self.push_event(&event);
            }
        }
    }

    fn drain_raw_json_lines(&mut self) {
        loop {
            let Some(index) = self.buffer.find('\n') else {
                return;
            };
            let line = self.buffer[..index].trim();
            if !line.starts_with('{') && line != "[DONE]" {
                return;
            }

            let event = line.to_string();
            self.buffer.drain(..index + 1);
            self.push_event(&event);
            if self.done {
                return;
            }
        }
    }

    fn push_event(&mut self, event: &str) {
        let Some(data) = event_data(event) else {
            return;
        };

        if data == "[DONE]" {
            self.done = true;
            return;
        }

        let mut value = match serde_json::from_str::<Value>(&data) {
            Ok(value) => value,
            Err(err) => {
                self.done = true;
                self.pending.push_back(Err(SigmaError::ProviderResponse {
                    provider: self.provider.clone(),
                    message: err.to_string(),
                }));
                return;
            }
        };

        if self.flavor.sanitizes_usage() {
            sanitize_null_usage_tokens(&mut value);
        }

        let chunk = serde_json::from_value(value).map_err(|err| SigmaError::ProviderResponse {
            provider: self.provider.clone(),
            message: err.to_string(),
        });
        self.pending.push_back(chunk);
    }
}

impl Stream for OpenAiSseStream {
    type Item = SigmaResult<CreateChatCompletionStreamResponse>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(item) = self.pending.pop_front() {
            return Poll::Ready(Some(item));
        }

        if self.done {
            return Poll::Ready(None);
        }

        loop {
            let poll = self.stream.as_mut().poll_next(cx);
            match poll {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.push_chunk(chunk);
                    if let Some(item) = self.pending.pop_front() {
                        return Poll::Ready(Some(item));
                    }
                    if self.done {
                        return Poll::Ready(None);
                    }
                }
                Poll::Ready(Some(Err(err))) => return Poll::Ready(Some(Err(err))),
                Poll::Ready(None) => {
                    self.drain_buffer(true);
                    self.done = true;
                    return Poll::Ready(self.pending.pop_front());
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

fn resolve_api_base<TConfig>(
    init: &ProviderInit<TConfig>,
    flavor: OpenAiFlavor,
) -> SigmaResult<String> {
    match flavor {
        OpenAiFlavor::OpenAi => Ok(init
            .common
            .api_base
            .clone()
            .or_else(|| non_empty_env("OPENAI_BASE_URL"))
            .or_else(|| non_empty_env("OPENAI_API_BASE"))
            .unwrap_or_else(|| OPENAI_DEFAULT_BASE_URL.to_string())),
        OpenAiFlavor::Compatible => init
            .common
            .api_base
            .clone()
            .or_else(|| non_empty_env("OPENAI_COMPATIBLE_API_BASE"))
            .or_else(|| non_empty_env("OPENAI_LIKE_API_BASE"))
            .ok_or_else(|| SigmaError::ProviderConfig {
                provider: Some(init.id.clone()),
                message: "openai-compatible provider requires api_base, OPENAI_COMPATIBLE_API_BASE, or OPENAI_LIKE_API_BASE".to_string(),
            }),
    }
}

fn resolve_api_key(api_key: Option<SecretString>, flavor: OpenAiFlavor) -> Option<SecretString> {
    api_key.or_else(|| match flavor {
        OpenAiFlavor::OpenAi => non_empty_env("OPENAI_API_KEY").map(SecretString::from),
        OpenAiFlavor::Compatible => non_empty_env("OPENAI_COMPATIBLE_API_KEY")
            .or_else(|| non_empty_env("OPENAI_LIKE_API_KEY"))
            .map(SecretString::from),
    })
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

fn header_map_from_config(
    provider: &ProviderId,
    headers: HashMap<String, String>,
) -> SigmaResult<HeaderMap> {
    let mut header_map = HeaderMap::new();

    for (name, value) in headers {
        let name =
            HeaderName::from_bytes(name.as_bytes()).map_err(|err| SigmaError::ProviderConfig {
                provider: Some(provider.clone()),
                message: format!("invalid header name `{name}`: {err}"),
            })?;
        let value = HeaderValue::from_str(&value).map_err(|err| SigmaError::ProviderConfig {
            provider: Some(provider.clone()),
            message: format!("invalid header value for `{name}`: {err}"),
        })?;
        header_map.insert(name, value);
    }

    Ok(header_map)
}

fn chat_completions_url(api_base: &str) -> String {
    let api_base = api_base.trim_end_matches('/');

    if api_base.ends_with("/chat/completions") {
        api_base.to_string()
    } else {
        format!("{api_base}/chat/completions")
    }
}

fn parse_response_json(provider: &ProviderId, body: &[u8]) -> SigmaResult<Value> {
    serde_json::from_slice(body).map_err(|err| SigmaError::ProviderResponse {
        provider: provider.clone(),
        message: err.to_string(),
    })
}

fn sanitize_null_usage_tokens(value: &mut Value) {
    let Some(usage) = value.get_mut("usage").and_then(Value::as_object_mut) else {
        return;
    };

    for (key, value) in usage {
        if key.ends_with("_tokens") && value.is_null() {
            *value = Value::from(0);
        }
    }
}

fn openai_error_response(
    context: &ChatAdapterContext<'_>,
    response: ProviderResponse,
) -> SigmaError {
    let body = serde_json::from_slice::<Value>(&response.body).ok();
    let error = body
        .as_ref()
        .and_then(|body| body.get("error"))
        .filter(|error| error.is_object());

    let code = error
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| fallback_error_message(response.status, &response.body));
    let details = error.cloned().or(body);

    SigmaError::ProviderBusiness {
        provider: context.provider.to_owned(),
        status: response.status,
        code,
        message,
        details,
    }
}

fn fallback_error_message(status: StatusCode, body: &[u8]) -> String {
    if body.is_empty() {
        status
            .canonical_reason()
            .unwrap_or("provider returned unsuccessful HTTP status")
            .to_string()
    } else {
        String::from_utf8_lossy(body).into_owned()
    }
}

fn event_data(event: &str) -> Option<String> {
    let mut data_lines = Vec::new();

    for line in event.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        } else if line.starts_with('{') || line == "[DONE]" {
            data_lines.push(line.to_string());
        }
    }

    if data_lines.is_empty() {
        None
    } else {
        Some(data_lines.join("\n"))
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
