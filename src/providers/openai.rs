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
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::config::{ChatParameterMap, SecretString};
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::types::chat::{
    AssistantContent, AssistantContentPart, ChatMessage, ChatResponse, ChatStreamChunk,
    DeveloperMessage, FileInput, SystemMessage, TextContent, ToolCall, ToolContent, UserContent,
    UserContentPart,
};
use crate::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatStream, ModelName,
    ProviderDriver, ProviderId, ProviderInit, ProviderKind, ProviderKindStatic, SigmaError,
    SigmaResult, StreamBehavior, submit_provider,
};

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

fn is_generated_body_key(key: &str) -> bool {
    key == "model" || key == "messages"
}

fn contains_provider_option(provider_options: Option<&ChatParameterMap>, key: &str) -> bool {
    provider_options.is_some_and(|provider_options| provider_options.contains_key(key))
}

fn rename_param(params: &mut ChatParameterMap, from: &str, to: &str) {
    if let Some(value) = params.remove(from) {
        params.insert(to.to_string(), value);
    }
}

fn openai_chat_body(
    provider: &ProviderId,
    provider_model: &ModelName,
    messages: &[ChatMessage],
    params: &ChatParameterMap,
    provider_options: Option<&ChatParameterMap>,
) -> SigmaResult<Value> {
    let mut body = Map::new();

    for (key, value) in params {
        if !is_generated_body_key(key.as_str())
            && !contains_provider_option(provider_options, key.as_str())
        {
            body.insert(key.clone(), value.clone());
        }
    }
    if !contains_provider_option(provider_options, "model") {
        body.insert(
            "model".to_string(),
            Value::String(provider_model.to_string()),
        );
    }
    if !contains_provider_option(provider_options, "messages") {
        body.insert("messages".to_string(), openai_messages(provider, messages)?);
    }
    if let Some(provider_options) = provider_options {
        for (key, value) in provider_options {
            body.insert(key.clone(), value.clone());
        }
    }

    Ok(Value::Object(body))
}

fn openai_messages(provider: &ProviderId, messages: &[ChatMessage]) -> SigmaResult<Value> {
    messages
        .iter()
        .map(|message| openai_message(provider, message))
        .collect::<SigmaResult<Vec<_>>>()
        .map(Value::Array)
}

fn openai_message(provider: &ProviderId, message: &ChatMessage) -> SigmaResult<Value> {
    let mut object = Map::new();
    match message {
        ChatMessage::Developer(message) => {
            insert_text_message(&mut object, "developer", message, provider)?;
        }
        ChatMessage::System(message) => {
            insert_text_message(&mut object, "system", message, provider)?;
        }
        ChatMessage::User(message) => {
            object.insert("role".to_string(), Value::String("user".to_string()));
            object.insert(
                "content".to_string(),
                openai_user_content(provider, &message.content)?,
            );
            insert_optional_string(&mut object, "name", message.name.as_deref());
        }
        ChatMessage::Assistant(message) => {
            object.insert("role".to_string(), Value::String("assistant".to_string()));
            if let Some(content) = &message.content {
                object.insert(
                    "content".to_string(),
                    openai_assistant_content(provider, content)?,
                );
            }
            insert_optional_string(&mut object, "refusal", message.refusal.as_deref());
            insert_optional_string(&mut object, "name", message.name.as_deref());
            if let Some(audio) = &message.audio {
                object.insert("audio".to_string(), serialized_value(provider, audio)?);
            }
            if let Some(tool_calls) = &message.tool_calls {
                object.insert(
                    "tool_calls".to_string(),
                    Value::Array(tool_calls.iter().map(openai_tool_call).collect::<Vec<_>>()),
                );
            }
        }
        ChatMessage::Tool(message) => {
            object.insert("role".to_string(), Value::String("tool".to_string()));
            object.insert(
                "content".to_string(),
                openai_tool_content(provider, &message.content)?,
            );
            object.insert(
                "tool_call_id".to_string(),
                Value::String(message.tool_call_id.clone()),
            );
        }
    }

    Ok(Value::Object(object))
}

fn insert_text_message<T>(
    object: &mut Map<String, Value>,
    role: &str,
    message: &T,
    provider: &ProviderId,
) -> SigmaResult<()>
where
    T: TextMessageFields,
{
    object.insert("role".to_string(), Value::String(role.to_string()));
    object.insert(
        "content".to_string(),
        openai_text_content(provider, message.content())?,
    );
    insert_optional_string(object, "name", message.name());
    Ok(())
}

trait TextMessageFields {
    fn content(&self) -> &TextContent;
    fn name(&self) -> Option<&str>;
}

impl TextMessageFields for DeveloperMessage {
    fn content(&self) -> &TextContent {
        &self.content
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

impl TextMessageFields for SystemMessage {
    fn content(&self) -> &TextContent {
        &self.content
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

fn openai_text_content(provider: &ProviderId, content: &TextContent) -> SigmaResult<Value> {
    match content {
        TextContent::Text(text) => Ok(Value::String(text.clone())),
        TextContent::Parts(parts) => parts
            .iter()
            .map(|part| openai_text_part(provider, part))
            .collect::<SigmaResult<Vec<_>>>()
            .map(Value::Array),
    }
}

fn openai_user_content(provider: &ProviderId, content: &UserContent) -> SigmaResult<Value> {
    match content {
        UserContent::Text(text) => Ok(Value::String(text.clone())),
        UserContent::Parts(parts) => parts
            .iter()
            .map(|part| openai_user_content_part(provider, part))
            .collect::<SigmaResult<Vec<_>>>()
            .map(Value::Array),
    }
}

fn openai_user_content_part(provider: &ProviderId, part: &UserContentPart) -> SigmaResult<Value> {
    match part {
        UserContentPart::Text(part) => openai_text_part(provider, part),
        UserContentPart::Image(part) => {
            let mut object = Map::new();
            object.insert("type".to_string(), Value::String("image_url".to_string()));
            object.insert(
                "image_url".to_string(),
                serialized_value(provider, &part.image)?,
            );
            insert_cache_control(&mut object, provider, part.cache_control.as_ref())?;
            Ok(Value::Object(object))
        }
        UserContentPart::Audio(part) => Ok(json!({
            "type": "input_audio",
            "input_audio": part.input_audio,
        })),
        UserContentPart::File(part) => {
            let mut object = Map::new();
            object.insert("type".to_string(), Value::String("file".to_string()));
            object.insert("file".to_string(), openai_file_input(provider, &part.file)?);
            insert_cache_control(&mut object, provider, part.cache_control.as_ref())?;
            Ok(Value::Object(object))
        }
    }
}

fn openai_assistant_content(
    provider: &ProviderId,
    content: &AssistantContent,
) -> SigmaResult<Value> {
    match content {
        AssistantContent::Text(text) => Ok(Value::String(text.clone())),
        AssistantContent::Parts(parts) => parts
            .iter()
            .map(|part| openai_assistant_content_part(provider, part))
            .collect::<SigmaResult<Vec<_>>>()
            .map(Value::Array),
    }
}

fn openai_assistant_content_part(
    provider: &ProviderId,
    part: &AssistantContentPart,
) -> SigmaResult<Value> {
    let mut object = Map::new();
    match part {
        AssistantContentPart::Text(part) => return openai_text_part(provider, part),
        AssistantContentPart::Refusal(part) => {
            object.insert("type".to_string(), Value::String("refusal".to_string()));
            object.insert("refusal".to_string(), Value::String(part.refusal.clone()));
        }
    }
    Ok(Value::Object(object))
}

fn openai_tool_content(provider: &ProviderId, content: &ToolContent) -> SigmaResult<Value> {
    match content {
        ToolContent::Text(text) => Ok(Value::String(text.clone())),
        ToolContent::Parts(parts) => parts
            .iter()
            .map(|part| openai_text_part(provider, part))
            .collect::<SigmaResult<Vec<_>>>()
            .map(Value::Array),
    }
}

fn openai_text_part(
    provider: &ProviderId,
    part: &crate::types::chat::TextPart,
) -> SigmaResult<Value> {
    let mut object = Map::new();
    object.insert("type".to_string(), Value::String("text".to_string()));
    object.insert("text".to_string(), Value::String(part.text.clone()));
    insert_cache_control(&mut object, provider, part.cache_control.as_ref())?;
    Ok(Value::Object(object))
}

fn openai_file_input(provider: &ProviderId, file: &FileInput) -> SigmaResult<Value> {
    let mut object = Map::new();
    insert_optional_string(&mut object, "file_data", file.data.as_deref());
    insert_optional_string(&mut object, "file_id", file.id.as_deref());
    insert_optional_string(&mut object, "filename", file.filename.as_deref());
    insert_optional_string(&mut object, "format", file.media_type.as_deref());
    if let Some(detail) = &file.detail {
        object.insert("detail".to_string(), serialized_value(provider, detail)?);
    }
    if let Some(video_metadata) = &file.video_metadata {
        object.insert(
            "video_metadata".to_string(),
            serialized_value(provider, video_metadata)?,
        );
    }
    Ok(Value::Object(object))
}

fn openai_tool_call(tool_call: &ToolCall) -> Value {
    match tool_call {
        ToolCall::Function(call) => json!({
            "type": "function",
            "id": call.id,
            "function": call.function,
        }),
        ToolCall::Custom(call) => json!({
            "type": "custom",
            "id": call.id,
            "custom_tool": call.custom_tool,
        }),
    }
}

fn insert_optional_string(object: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        object.insert(key.to_string(), Value::String(value.to_string()));
    }
}

fn insert_cache_control(
    object: &mut Map<String, Value>,
    provider: &ProviderId,
    cache_control: Option<&crate::types::chat::CacheControl>,
) -> SigmaResult<()> {
    if let Some(cache_control) = cache_control {
        object.insert(
            "cache_control".to_string(),
            serialized_value(provider, cache_control)?,
        );
    }
    Ok(())
}

fn serialized_value<T: Serialize>(provider: &ProviderId, value: &T) -> SigmaResult<Value> {
    serde_json::to_value(value).map_err(|err| SigmaError::ProviderTransform {
        provider: provider.clone(),
        message: err.to_string(),
    })
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

struct OpenAiSseStream {
    provider: ProviderId,
    stream: ProviderByteStream,
    buffer: String,
    pending: VecDeque<SigmaResult<ChatStreamChunk>>,
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
        map_stream_reasoning_content(&mut value);

        let chunk = serde_json::from_value(value).map_err(|err| SigmaError::ProviderResponse {
            provider: self.provider.clone(),
            message: err.to_string(),
        });
        self.pending.push_back(chunk);
    }
}

impl Stream for OpenAiSseStream {
    type Item = SigmaResult<ChatStreamChunk>;

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

fn map_response_reasoning_content(value: &mut Value) {
    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };

    for choice in choices {
        if let Some(message) = choice.get_mut("message").and_then(Value::as_object_mut) {
            move_reasoning_content(message);
        }
    }
}

fn map_stream_reasoning_content(value: &mut Value) {
    let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };

    for choice in choices {
        if let Some(delta) = choice.get_mut("delta").and_then(Value::as_object_mut) {
            move_reasoning_content(delta);
        }
    }
}

fn move_reasoning_content(object: &mut Map<String, Value>) {
    let Some(reasoning_content) = object.remove("reasoning_content") else {
        return;
    };
    let Some(reasoning_text) = reasoning_content.as_str().filter(|value| !value.is_empty()) else {
        return;
    };
    let reasoning_block = json!({
        "type": "text",
        "text": reasoning_text,
    });

    match object.get_mut("reasoning").and_then(Value::as_array_mut) {
        Some(reasoning) => reasoning.push(reasoning_block),
        None => {
            object.insert("reasoning".to_string(), Value::Array(vec![reasoning_block]));
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
