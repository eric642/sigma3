use std::collections::{BTreeMap, HashMap, VecDeque};
use std::env;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use http::header::{AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use serde::Deserialize;
use serde_json::{Map, Value, json};

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
    fn from_openai_config(init: ProviderInit) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let _config: OpenAiConfig = init.deserialize_config()?;

        Self::from_config(init, OpenAiFlavor::OpenAi, RequestFieldRules::default())
    }

    fn from_compatible_config(init: ProviderInit) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let config: OpenAiCompatibleConfig = init.deserialize_config()?;
        let request_field_rules =
            RequestFieldRules::from_config(config.request_field_rules, &init.id)?;

        Self::from_config(init, OpenAiFlavor::Compatible, request_field_rules)
    }

    fn from_config(
        init: ProviderInit,
        flavor: OpenAiFlavor,
        request_field_rules: RequestFieldRules,
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
                request_field_rules,
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

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OpenAiConfig {}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OpenAiCompatibleConfig {
    request_field_rules: RequestFieldRulesConfig,
}

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
        self == Self::Compatible
    }
}

struct OpenAiChatAdapter {
    provider: ProviderId,
    api_base: String,
    api_key: Option<SecretString>,
    headers: HeaderMap,
    flavor: OpenAiFlavor,
    request_field_rules: RequestFieldRules,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RequestFieldRulesConfig {
    #[serde(rename = "map")]
    mappings: BTreeMap<String, String>,
    remove: Vec<String>,
    models: BTreeMap<String, RequestFieldRuleSetConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RequestFieldRuleSetConfig {
    #[serde(rename = "map")]
    mappings: BTreeMap<String, String>,
    remove: Vec<String>,
}

#[derive(Debug, Default)]
struct RequestFieldRules {
    global: RequestFieldRuleSet,
    models: HashMap<ModelName, RequestFieldRuleSet>,
}

impl RequestFieldRules {
    fn from_config(config: RequestFieldRulesConfig, provider: &ProviderId) -> SigmaResult<Self> {
        let global = RequestFieldRuleSet::from_parts(
            config.mappings,
            config.remove,
            provider,
            "request_field_rules",
        )?;
        let mut models = HashMap::new();

        for (model, rules) in config.models {
            let context = format!("request_field_rules.models.{model}");
            let rules =
                RequestFieldRuleSet::from_parts(rules.mappings, rules.remove, provider, &context)?;
            models.insert(ModelName::from(model), rules);
        }

        Ok(Self { global, models })
    }

    fn apply(&self, provider_model: &ModelName, body: &mut Value) {
        self.global.apply(body);
        if let Some(model_rules) = self.models.get(provider_model) {
            model_rules.apply(body);
        }
    }

    fn extend_supported_openai_params(&self, params: &mut Vec<String>) {
        self.global.extend_supported_openai_params(params);
        for model_rules in self.models.values() {
            model_rules.extend_supported_openai_params(params);
        }
    }
}

#[derive(Debug, Default)]
struct RequestFieldRuleSet {
    mappings: Vec<RequestFieldMapping>,
    removals: Vec<JsonPointer>,
}

impl RequestFieldRuleSet {
    fn from_parts(
        mappings: BTreeMap<String, String>,
        removals: Vec<String>,
        provider: &ProviderId,
        context: &str,
    ) -> SigmaResult<Self> {
        let mut parsed_mappings = Vec::with_capacity(mappings.len());
        for (source, target) in mappings {
            parsed_mappings.push(RequestFieldMapping {
                source: JsonPointer::parse(&source, provider, context)?,
                target: JsonPointer::parse(&target, provider, context)?,
            });
        }

        let mut parsed_removals = Vec::with_capacity(removals.len());
        for removal in removals {
            parsed_removals.push(JsonPointer::parse(&removal, provider, context)?);
        }

        Ok(Self {
            mappings: parsed_mappings,
            removals: parsed_removals,
        })
    }

    fn apply(&self, body: &mut Value) {
        for mapping in &self.mappings {
            move_json_pointer(body, &mapping.source, &mapping.target);
        }

        for removal in &self.removals {
            remove_json_pointer(body, removal);
        }
    }

    fn extend_supported_openai_params(&self, params: &mut Vec<String>) {
        for mapping in &self.mappings {
            push_supported_param(params, mapping.source.top_level_field());
        }

        for removal in &self.removals {
            push_supported_param(params, removal.top_level_field());
        }
    }
}

#[derive(Debug)]
struct RequestFieldMapping {
    source: JsonPointer,
    target: JsonPointer,
}

#[derive(Debug)]
struct JsonPointer {
    tokens: Vec<String>,
}

impl JsonPointer {
    fn parse(pointer: &str, provider: &ProviderId, context: &str) -> SigmaResult<Self> {
        if pointer.is_empty() || !pointer.starts_with('/') {
            return Err(invalid_json_pointer(
                provider,
                pointer,
                context,
                "pointers must start with `/`",
            ));
        }

        let mut tokens = Vec::new();
        for token in pointer[1..].split('/') {
            tokens
                .push(decode_json_pointer_token(token).map_err(|message| {
                    invalid_json_pointer(provider, pointer, context, message)
                })?);
        }

        Ok(Self { tokens })
    }

    fn top_level_field(&self) -> &str {
        self.tokens.first().map(String::as_str).unwrap_or_default()
    }
}

fn decode_json_pointer_token(token: &str) -> Result<String, String> {
    let mut decoded = String::with_capacity(token.len());
    let mut chars = token.chars();

    while let Some(ch) = chars.next() {
        if ch != '~' {
            decoded.push(ch);
            continue;
        }

        match chars.next() {
            Some('0') => decoded.push('~'),
            Some('1') => decoded.push('/'),
            Some(other) => return Err(format!("invalid escape `~{other}`")),
            None => return Err("invalid trailing `~` escape".to_string()),
        }
    }

    Ok(decoded)
}

fn invalid_json_pointer(
    provider: &ProviderId,
    pointer: &str,
    context: &str,
    message: impl Into<String>,
) -> SigmaError {
    SigmaError::ProviderConfig {
        provider: Some(provider.clone()),
        message: format!(
            "invalid JSON Pointer `{pointer}` in openai-compatible config {context}: {}",
            message.into()
        ),
    }
}

fn push_supported_param(params: &mut Vec<String>, param: &str) {
    if !param.is_empty() && !params.iter().any(|existing| existing == param) {
        params.push(param.to_string());
    }
}

fn build_openai_chat_body(
    params: &ChatParameterMap,
    provider_model: &ModelName,
    messages: &Value,
    body_overrides: Option<&ChatParameterMap>,
    request_field_rules: &RequestFieldRules,
) -> Value {
    let mut body = Map::new();
    for (key, value) in params {
        if !is_generated_body_key(key.as_str()) {
            body.insert(key.clone(), value.clone());
        }
    }
    body.insert(
        "model".to_string(),
        Value::String(provider_model.as_str().to_string()),
    );
    body.insert("messages".to_string(), messages.clone());

    let mut body = Value::Object(body);
    request_field_rules.apply(provider_model, &mut body);

    if let (Value::Object(body), Some(body_overrides)) = (&mut body, body_overrides) {
        for (key, value) in body_overrides {
            body.insert(key.clone(), value.clone());
        }
    }

    body
}

fn move_json_pointer(body: &mut Value, source: &JsonPointer, target: &JsonPointer) {
    if let Some(value) = remove_json_pointer(body, source) {
        set_json_pointer(body, target, value);
    }
}

fn remove_json_pointer(body: &mut Value, pointer: &JsonPointer) -> Option<Value> {
    let (last, parents) = pointer.tokens.split_last()?;
    let parent = json_pointer_parent_mut(body, parents)?;

    match parent {
        Value::Object(object) => object.remove(last),
        Value::Array(array) => {
            let index = array_index(last)?;
            if index < array.len() {
                Some(array.remove(index))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn set_json_pointer(body: &mut Value, pointer: &JsonPointer, value: Value) -> bool {
    let Some((last, parents)) = pointer.tokens.split_last() else {
        return false;
    };
    let Some(parent) = json_pointer_parent_mut_or_create(body, parents) else {
        return false;
    };

    match parent {
        Value::Object(object) => {
            object.insert(last.clone(), value);
            true
        }
        Value::Array(array) => {
            let Some(index) = array_index(last) else {
                return false;
            };
            let Some(slot) = array.get_mut(index) else {
                return false;
            };
            *slot = value;
            true
        }
        _ => false,
    }
}

fn json_pointer_parent_mut<'a>(
    mut value: &'a mut Value,
    tokens: &[String],
) -> Option<&'a mut Value> {
    for token in tokens {
        value = match value {
            Value::Object(object) => object.get_mut(token)?,
            Value::Array(array) => array.get_mut(array_index(token)?)?,
            _ => return None,
        };
    }

    Some(value)
}

fn json_pointer_parent_mut_or_create<'a>(
    mut value: &'a mut Value,
    tokens: &[String],
) -> Option<&'a mut Value> {
    for token in tokens {
        value = match value {
            Value::Object(object) => object
                .entry(token.clone())
                .or_insert_with(|| Value::Object(Map::new())),
            Value::Array(array) => array.get_mut(array_index(token)?)?,
            _ => return None,
        };
    }

    Some(value)
}

fn array_index(token: &str) -> Option<usize> {
    token.parse::<usize>().ok()
}

fn is_generated_body_key(key: &str) -> bool {
    key == "model" || key == "messages"
}

impl ChatCompletionAdapter for OpenAiChatAdapter {
    fn supported_openai_params(&self) -> Vec<String> {
        let mut params = SUPPORTED_OPENAI_CHAT_PARAMS
            .iter()
            .map(|param| (*param).to_string())
            .collect::<Vec<_>>();
        self.request_field_rules
            .extend_supported_openai_params(&mut params);
        params
    }

    fn translate_messages(&self, messages: &[ChatCompletionRequestMessage]) -> SigmaResult<Value> {
        serde_json::to_value(messages).map_err(|err| SigmaError::ProviderTransform {
            provider: self.provider.clone(),
            message: err.to_string(),
        })
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
        let body = build_openai_chat_body(
            &request.params,
            request.context.provider_model,
            &request.messages,
            request.body_overrides,
            &self.request_field_rules,
        );
        let body = serde_json::to_vec(&body).map_err(|err| SigmaError::ProviderTransform {
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
            body: Bytes::from(body),
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

fn resolve_api_base(init: &ProviderInit, flavor: OpenAiFlavor) -> SigmaResult<String> {
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

fn openai_config_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "default": {},
        "description": "The built-in OpenAI provider does not require provider-specific config."
    })
}

fn openai_compatible_config_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "default": {},
        "properties": {
            "request_field_rules": {
                "type": "object",
                "additionalProperties": false,
                "default": {},
                "description": "Explicit request-body field mapping and removal rules for OpenAI-compatible endpoints. Rules use JSON Pointer paths and run before request metadata overrides.",
                "properties": {
                    "map": request_field_rule_map_schema(),
                    "remove": request_field_rule_remove_schema(),
                    "models": {
                        "type": "object",
                        "additionalProperties": request_field_rule_set_schema(),
                        "default": {},
                        "description": "Rules keyed by provider-native model name. Matching model rules run after provider-level rules."
                    }
                }
            }
        }
    })
}

fn request_field_rule_set_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "default": {},
        "properties": {
            "map": request_field_rule_map_schema(),
            "remove": request_field_rule_remove_schema()
        }
    })
}

fn request_field_rule_map_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": {
            "type": "string",
            "pattern": "^/"
        },
        "propertyNames": {
            "pattern": "^/"
        },
        "default": {},
        "description": "Moves each source JSON Pointer value to the target JSON Pointer. Sources are removed and targets are overwritten when present."
    })
}

fn request_field_rule_remove_schema() -> Value {
    json!({
        "type": "array",
        "items": {
            "type": "string",
            "pattern": "^/"
        },
        "default": [],
        "description": "JSON Pointer paths to remove from the provider request body. Missing paths are ignored."
    })
}

submit_provider! {
    kind: OPENAI_KIND,
    constructor: OpenAiProvider::from_openai_config,
    config_schema: openai_config_schema,
}

submit_provider! {
    kind: OPENAI_COMPATIBLE_KIND,
    constructor: OpenAiProvider::from_compatible_config,
    config_schema: openai_compatible_config_schema,
}
