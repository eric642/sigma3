use std::collections::{BTreeSet, HashMap};
use std::env;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use futures_core::Stream;
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::config::{ChatParameterMap, SecretString};
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::types::chat::{
    Annotation, AssistantContent, AssistantContentPart, AssistantDelta, AssistantMessage,
    CacheControl, CacheControlTtl, CacheControlType, ChatChoice, ChatMessage, ChatResponse,
    ChatResponseMessage, ChatStreamChoice, ChatStreamChunk, FilePart, FinishReason,
    FunctionCallDelta, FunctionToolCall, HostedToolUsage, ImagePart, ProviderContextBlock,
    ReasoningBlock, Role, TextContent, ToolCall, ToolCallDelta, ToolCallKind, ToolContent,
    ToolMessage, UrlCitation, Usage, UserContent, UserContentPart,
};
use crate::types::shared::{
    AnthropicThinkingParam, AnthropicThinkingType, CompletionTokensDetails, FunctionCall,
    PromptTokensDetails, ResponseFormat,
};
use crate::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatStream, ModelName,
    ProviderDriver, ProviderId, ProviderInit, ProviderKind, ProviderKindStatic, SigmaError,
    SigmaResult, StreamBehavior, submit_provider,
};

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

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
struct AnthropicConfig {
    /// Anthropic API version header value.
    anthropic_version: Option<String>,
    /// Default `max_tokens` used when the request omits both token limit fields.
    default_max_tokens: Option<u32>,
    /// Optional bearer auth token. Prefer `api_key` for normal Anthropic API keys.
    auth_token: Option<SecretString>,
    /// Static Anthropic beta header values added to every request.
    beta: Vec<String>,
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

impl AnthropicChatAdapter {
    fn collect_beta_values(
        &self,
        params: &mut ChatParameterMap,
        provider_options: Option<&ChatParameterMap>,
    ) -> BTreeSet<String> {
        let mut beta_values = self.beta.iter().cloned().collect::<BTreeSet<_>>();
        if let Some(value) = params.remove("anthropic_beta") {
            insert_beta_value(&mut beta_values, &value);
        }
        if let Some(value) = provider_options.and_then(|options| options.get("anthropic_beta")) {
            insert_beta_value(&mut beta_values, value);
        }
        beta_values
    }
}

fn insert_beta_value(beta_values: &mut BTreeSet<String>, value: &Value) {
    match value {
        Value::String(value) => {
            insert_split_beta(beta_values, value);
        }
        Value::Array(values) => {
            for value in values {
                if let Some(value) = value.as_str() {
                    insert_split_beta(beta_values, value);
                }
            }
        }
        _ => {}
    }
}

#[derive(Default)]
struct ToolNameMaps {
    forward: HashMap<String, String>,
    reverse: HashMap<String, String>,
}

impl ToolNameMaps {
    fn has_rewrites(&self) -> bool {
        !self.forward.is_empty()
    }
}

struct TranslatedMessages {
    messages: Vec<Value>,
    system: Vec<Value>,
}

fn prepare_tools(params: &mut ChatParameterMap) -> SigmaResult<ToolNameMaps> {
    let Some(value) = params.get_mut("tools") else {
        return Ok(ToolNameMaps::default());
    };
    let Some(tools) = value.as_array_mut() else {
        return Ok(ToolNameMaps::default());
    };

    for tool in tools.iter_mut() {
        if tool.get("input_schema").is_some() {
            continue;
        }
        let mapped = map_openai_tool(tool)?;
        *tool = mapped;
    }

    let names = tools
        .iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("custom"))
        .filter_map(|tool| tool.get("name").and_then(Value::as_str))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let maps = build_tool_name_maps(&names);
    if maps.has_rewrites() {
        for tool in tools {
            let Some(name) = tool.get("name").and_then(Value::as_str) else {
                continue;
            };
            if let Some(mapped) = maps.forward.get(name).cloned()
                && let Some(object) = tool.as_object_mut()
            {
                object.insert("name".to_string(), Value::String(mapped));
            }
        }
    }

    Ok(maps)
}

fn map_openai_tool(tool: &Value) -> SigmaResult<Value> {
    let Some(object) = tool.as_object() else {
        return Ok(tool.clone());
    };
    let tool_type = object.get("type").and_then(Value::as_str);
    if tool_type == Some("function") {
        let function = object
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| SigmaError::ProviderTransform {
                provider: ProviderId::from("anthropic"),
                message: "function tool is missing function object".to_string(),
            })?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("tool")
            .to_string();
        let mut mapped = Map::new();
        mapped.insert("type".to_string(), Value::String("custom".to_string()));
        mapped.insert("name".to_string(), Value::String(name));
        if let Some(description) = function.get("description") {
            mapped.insert("description".to_string(), description.clone());
        }

        let mut input_schema = function
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
        if input_schema.get("type").and_then(Value::as_str) != Some("object")
            && let Some(input_schema) = input_schema.as_object_mut()
        {
            input_schema.insert("type".to_string(), Value::String("object".to_string()));
            input_schema
                .entry("properties")
                .or_insert_with(|| Value::Object(Map::new()));
        }
        mapped.insert("input_schema".to_string(), input_schema);
        Ok(Value::Object(mapped))
    } else {
        Ok(tool.clone())
    }
}

fn build_tool_name_maps(names: &[String]) -> ToolNameMaps {
    let mut forward = HashMap::new();
    let mut used = BTreeSet::new();

    for original in names {
        let candidate = sanitize_tool_name(original);
        if candidate == *original {
            used.insert(candidate);
        }
    }

    for original in names {
        let candidate = sanitize_tool_name(original);
        if candidate == *original || forward.contains_key(original) {
            continue;
        }

        let mut unique = candidate.clone();
        let mut suffix = 1;
        while used.contains(&unique) {
            suffix += 1;
            let suffix_value = format!("_{suffix}");
            let max_head = 128usize.saturating_sub(suffix_value.len());
            unique = format!("{}{}", truncate_chars(&candidate, max_head), suffix_value);
        }
        forward.insert(original.clone(), unique.clone());
        used.insert(unique);
    }

    let reverse = forward
        .iter()
        .map(|(original, mapped)| (mapped.clone(), original.clone()))
        .collect();

    ToolNameMaps { forward, reverse }
}

fn sanitize_tool_name(name: &str) -> String {
    truncate_chars(
        &name
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .collect::<String>(),
        128,
    )
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn apply_tool_choice_name_map(params: &mut ChatParameterMap, forward: &HashMap<String, String>) {
    let Some(tool_choice) = params.get_mut("tool_choice") else {
        return;
    };
    if let Some(object) = tool_choice.as_object_mut() {
        if let Some(name) = object.get("name").and_then(Value::as_str) {
            if let Some(mapped) = forward.get(name).cloned() {
                object.insert("name".to_string(), Value::String(mapped));
            }
        } else if let Some(function) = object.get_mut("function").and_then(Value::as_object_mut)
            && let Some(name) = function.get("name").and_then(Value::as_str)
            && let Some(mapped) = forward.get(name).cloned()
        {
            function.insert("name".to_string(), Value::String(mapped));
        }
    }
}

fn map_token_params(params: &mut ChatParameterMap, default_max_tokens: u32) {
    if let Some(value) = params.remove("max_completion_tokens") {
        params.entry("max_tokens".to_string()).or_insert(value);
    }
    params
        .entry("max_tokens".to_string())
        .or_insert_with(|| Value::from(default_max_tokens));
}

fn map_stop_sequences(params: &mut ChatParameterMap) {
    let Some(value) = params.remove("stop") else {
        return;
    };
    match value {
        Value::String(value) => {
            params.insert(
                "stop_sequences".to_string(),
                Value::Array(vec![Value::String(value)]),
            );
        }
        Value::Array(value) => {
            params.insert("stop_sequences".to_string(), Value::Array(value));
        }
        _ => {}
    }
}

fn map_reasoning_effort(params: &mut ChatParameterMap, model: &ModelName) -> SigmaResult<()> {
    if params.contains_key("thinking") {
        params.remove("reasoning_effort");
        return Ok(());
    }
    let Some(value) = params.remove("reasoning_effort") else {
        return Ok(());
    };
    let Some(value) = value.as_str() else {
        return Ok(());
    };
    let Some(thinking) = thinking_for_reasoning_effort(value, model.as_str())? else {
        return Ok(());
    };
    params.insert(
        "thinking".to_string(),
        serde_json::to_value(thinking).map_err(|err| SigmaError::ProviderTransform {
            provider: ProviderId::from("anthropic"),
            message: err.to_string(),
        })?,
    );
    Ok(())
}

fn thinking_for_reasoning_effort(
    value: &str,
    model: &str,
) -> SigmaResult<Option<AnthropicThinkingParam>> {
    if value == "none" {
        return Ok(None);
    }
    if is_adaptive_thinking_model(model) {
        return Ok(Some(AnthropicThinkingParam {
            r#type: AnthropicThinkingType::Adaptive,
            budget_tokens: None,
        }));
    }

    let budget_tokens = match value {
        "minimal" => 1024,
        "low" => 1024,
        "medium" => 2048,
        "high" => 4096,
        "xhigh" => 8192,
        "max" => 16384,
        other => {
            return Err(SigmaError::ProviderTransform {
                provider: ProviderId::from("anthropic"),
                message: format!("unsupported reasoning_effort `{other}` for anthropic provider"),
            });
        }
    };

    Ok(Some(AnthropicThinkingParam {
        r#type: AnthropicThinkingType::Enabled,
        budget_tokens: Some(budget_tokens),
    }))
}

fn is_adaptive_thinking_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    [
        "opus-4-6",
        "opus_4_6",
        "opus-4.6",
        "opus_4.6",
        "sonnet-4-6",
        "sonnet_4_6",
        "sonnet-4.6",
        "sonnet_4.6",
        "opus-4-7",
        "opus_4_7",
        "opus-4.7",
        "opus_4.7",
    ]
    .iter()
    .any(|needle| model.contains(needle))
}

fn map_user_metadata(params: &mut ChatParameterMap) {
    let Some(value) = params.remove("user") else {
        return;
    };
    let Some(user_id) = value.as_str().filter(|value| valid_user_id(value)) else {
        return;
    };
    params.insert("metadata".to_string(), json!({ "user_id": user_id }));
}

fn valid_user_id(value: &str) -> bool {
    !looks_like_email(value) && !looks_like_phone(value)
}

fn looks_like_email(value: &str) -> bool {
    let Some((left, right)) = value.split_once('@') else {
        return false;
    };
    !left.is_empty() && right.contains('.') && !right.ends_with('.')
}

fn looks_like_phone(value: &str) -> bool {
    let digit_count = value.chars().filter(|ch| ch.is_ascii_digit()).count();
    digit_count >= 7
        && value
            .chars()
            .all(|ch| ch.is_ascii_digit() || matches!(ch, '+' | ' ' | '(' | ')' | '-'))
}

fn map_response_format(params: &mut ChatParameterMap, model: &ModelName) -> SigmaResult<()> {
    if params.contains_key("output_format") {
        params.remove("response_format");
        return Ok(());
    }
    let Some(value) = params.remove("response_format") else {
        return Ok(());
    };
    let format = serde_json::from_value::<ResponseFormat>(value).map_err(|err| {
        SigmaError::ProviderTransform {
            provider: ProviderId::from("anthropic"),
            message: err.to_string(),
        }
    })?;
    if supports_structured_output(model.as_str()) {
        let output_format = match format {
            ResponseFormat::Text => return Ok(()),
            ResponseFormat::JsonObject => json!({
                "type": "json_schema",
                "schema": {"type": "object", "additionalProperties": true, "properties": {}}
            }),
            ResponseFormat::JsonSchema { json_schema } => json!({
                "type": "json_schema",
                "schema": json_schema.schema.unwrap_or_else(|| json!({
                    "type": "object",
                    "additionalProperties": true,
                    "properties": {}
                }))
            }),
        };
        params.insert("output_format".to_string(), output_format);
        return Ok(());
    }

    let input_schema = match format {
        ResponseFormat::Text => return Ok(()),
        ResponseFormat::JsonObject => {
            json!({"type": "object", "additionalProperties": true, "properties": {}})
        }
        ResponseFormat::JsonSchema { json_schema } => json_schema.schema.unwrap_or_else(
            || json!({"type": "object", "additionalProperties": true, "properties": {}}),
        ),
    };
    let tool = json!({
        "type": "custom",
        "name": RESPONSE_FORMAT_TOOL_NAME,
        "input_schema": input_schema
    });
    add_tool(params, tool);
    if !params.contains_key("tool_choice") && !params.contains_key("thinking") {
        params.insert(
            "tool_choice".to_string(),
            json!({"type": "tool", "name": RESPONSE_FORMAT_TOOL_NAME}),
        );
    }
    Ok(())
}

fn provider_options_contain(provider_options: Option<&ChatParameterMap>, key: &str) -> bool {
    provider_options.is_some_and(|provider_options| provider_options.contains_key(key))
}

fn supports_structured_output(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    [
        "sonnet-4.5",
        "sonnet-4-5",
        "opus-4.1",
        "opus-4-1",
        "opus-4.5",
        "opus-4-5",
        "opus-4.6",
        "opus-4-6",
        "opus-4.7",
        "opus-4-7",
        "sonnet-4.6",
        "sonnet-4-6",
        "sonnet_4.6",
        "sonnet_4_6",
    ]
    .iter()
    .any(|needle| model.contains(needle))
}

fn add_tool(params: &mut ChatParameterMap, tool: Value) {
    match params.get_mut("tools").and_then(Value::as_array_mut) {
        Some(tools) => tools.push(tool),
        None => {
            params.insert("tools".to_string(), Value::Array(vec![tool]));
        }
    }
}

fn map_tool_choice(params: &mut ChatParameterMap) {
    let parallel_tool_calls = params
        .remove("parallel_tool_calls")
        .and_then(|value| value.as_bool());
    let tool_choice = params.remove("tool_choice");

    let mut mapped = match tool_choice {
        Some(Value::String(value)) if value == "auto" => Some(json!({"type": "auto"})),
        Some(Value::String(value)) if value == "required" => Some(json!({"type": "any"})),
        Some(Value::String(value)) if value == "none" => Some(json!({"type": "none"})),
        Some(Value::Object(mut object)) => {
            let tool_type = object.get("type").and_then(Value::as_str);
            if tool_type == Some("auto") {
                Some(json!({"type": "auto"}))
            } else if matches!(tool_type, Some("required" | "any")) {
                Some(json!({"type": "any"}))
            } else if tool_type == Some("none") {
                Some(json!({"type": "none"}))
            } else if tool_type == Some("tool") && object.get("name").is_some() {
                Some(Value::Object(object))
            } else if let Some(name) = object
                .get("function")
                .and_then(Value::as_object)
                .and_then(|function| function.get("name"))
                .cloned()
            {
                let mut choice = Map::new();
                choice.insert("type".to_string(), Value::String("tool".to_string()));
                choice.insert("name".to_string(), name);
                Some(Value::Object(choice))
            } else {
                object.insert("type".to_string(), Value::String("auto".to_string()));
                Some(Value::Object(object))
            }
        }
        Some(value) => Some(value),
        None if parallel_tool_calls.is_some() => Some(json!({"type": "auto"})),
        None => None,
    };

    if let Some(choice) = mapped.as_mut()
        && parallel_tool_calls.is_some()
        && choice.get("type").and_then(Value::as_str) != Some("none")
        && let Some(object) = choice.as_object_mut()
    {
        object.insert(
            "disable_parallel_tool_use".to_string(),
            Value::Bool(!parallel_tool_calls.unwrap_or(true)),
        );
    }

    if let Some(mapped) = mapped {
        params.insert("tool_choice".to_string(), mapped);
    }
}

fn map_web_search_tool(params: &mut ChatParameterMap) {
    let Some(value) = params.remove("web_search") else {
        return;
    };
    let user_location = value
        .get("user_location")
        .and_then(|location| location.get("approximate"))
        .cloned();
    let mut tool = Map::new();
    tool.insert(
        "type".to_string(),
        Value::String("web_search_20250305".to_string()),
    );
    tool.insert("name".to_string(), Value::String("web_search".to_string()));
    if let Some(user_location) = user_location {
        tool.insert(
            "user_location".to_string(),
            json!({
                "type": "approximate",
                "city": user_location.get("city").cloned().unwrap_or(Value::Null),
                "country": user_location.get("country").cloned().unwrap_or(Value::Null),
                "region": user_location.get("region").cloned().unwrap_or(Value::Null),
                "timezone": user_location.get("timezone").cloned().unwrap_or(Value::Null),
            }),
        );
    }
    add_tool(params, Value::Object(tool));
}

fn filter_metadata(params: &mut ChatParameterMap) {
    let Some(metadata) = params.get_mut("metadata").and_then(Value::as_object_mut) else {
        return;
    };
    let user_id = metadata.get("user_id").cloned();
    metadata.clear();
    if let Some(user_id) = user_id {
        metadata.insert("user_id".to_string(), user_id);
    }
}

fn is_internal_param(key: &str) -> bool {
    matches!(key, "anthropic_beta" | "reasoning_effort")
}

fn translate_anthropic_messages(
    provider: &ProviderId,
    messages: &[ChatMessage],
    tool_name_forward_map: &HashMap<String, String>,
) -> SigmaResult<TranslatedMessages> {
    let mut output = Vec::<Value>::new();
    let mut system = Vec::<Value>::new();

    for message in messages {
        match message {
            ChatMessage::Developer(message) => {
                append_system_content(&mut system, &developer_content_to_value(&message.content));
            }
            ChatMessage::System(message) => {
                append_system_content(&mut system, &system_content_to_value(&message.content));
            }
            ChatMessage::User(message) => {
                let content = user_content_blocks(provider, &message.content)?;
                append_anthropic_message(&mut output, "user", content);
            }
            ChatMessage::Tool(message) => {
                append_anthropic_message(&mut output, "user", vec![tool_result_block(message)]);
            }
            ChatMessage::Assistant(message) => {
                let content = assistant_content_blocks(provider, message, tool_name_forward_map)?;
                if !content.is_empty() {
                    append_anthropic_message(&mut output, "assistant", content);
                }
            }
        }
    }

    if output.is_empty() {
        append_anthropic_message(
            &mut output,
            "user",
            vec![json!({"type": "text", "text": "Please continue."})],
        );
    }
    trim_final_assistant_text(&mut output);

    Ok(TranslatedMessages {
        messages: output,
        system,
    })
}

fn append_system_content(system: &mut Vec<Value>, content: &[Value]) {
    for item in content {
        let text = item.get("text").and_then(Value::as_str).unwrap_or_default();
        if text.is_empty() || text.starts_with("x-anthropic-billing-header:") {
            continue;
        }
        system.push(item.clone());
    }
}

fn developer_content_to_value(content: &TextContent) -> Vec<Value> {
    match content {
        TextContent::Text(text) => {
            vec![json!({"type": "text", "text": text})]
        }
        TextContent::Parts(parts) => parts
            .iter()
            .map(|part| text_content_block(&part.text, part.cache_control.as_ref()))
            .collect(),
    }
}

fn system_content_to_value(content: &TextContent) -> Vec<Value> {
    match content {
        TextContent::Text(text) => {
            vec![json!({"type": "text", "text": text})]
        }
        TextContent::Parts(parts) => parts
            .iter()
            .map(|part| text_content_block(&part.text, part.cache_control.as_ref()))
            .collect(),
    }
}

fn user_content_blocks(provider: &ProviderId, content: &UserContent) -> SigmaResult<Vec<Value>> {
    match content {
        UserContent::Text(text) => Ok(vec![json!({"type": "text", "text": non_empty_text(text)})]),
        UserContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                UserContentPart::Text(part) => Ok(text_content_block(
                    non_empty_text(&part.text),
                    part.cache_control.as_ref(),
                )),
                UserContentPart::Image(part) => image_content_block(provider, part),
                UserContentPart::File(part) => file_content_block(provider, part),
                UserContentPart::Audio(_) => Err(SigmaError::ProviderTransform {
                    provider: provider.clone(),
                    message: "anthropic provider does not support input_audio message parts"
                        .to_string(),
                }),
            })
            .collect(),
    }
}

fn assistant_content_blocks(
    provider: &ProviderId,
    message: &AssistantMessage,
    tool_name_forward_map: &HashMap<String, String>,
) -> SigmaResult<Vec<Value>> {
    let mut content = Vec::new();
    content.extend(provider_context_content_blocks(
        provider,
        message.provider_context.as_deref(),
        is_compaction_block,
    ));
    if let Some(reasoning) = &message.reasoning {
        content.extend(reasoning.iter().filter_map(reasoning_block_to_anthropic));
    }
    if let Some(message_content) = &message.content {
        match message_content {
            AssistantContent::Text(text) => {
                if !text.is_empty() {
                    content.push(json!({"type": "text", "text": text}));
                }
            }
            AssistantContent::Parts(parts) => {
                for part in parts {
                    match part {
                        AssistantContentPart::Text(part) => {
                            if !part.text.is_empty() {
                                content.push(text_content_block(
                                    &part.text,
                                    part.cache_control.as_ref(),
                                ));
                            }
                        }
                        AssistantContentPart::Refusal(part) => {
                            if !part.refusal.is_empty() {
                                content.push(json!({"type": "text", "text": part.refusal}));
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(tool_calls) = &message.tool_calls {
        for tool_call in tool_calls {
            if let Some(tool_use) =
                tool_call_to_anthropic(provider, tool_call, tool_name_forward_map)?
            {
                content.push(tool_use);
            }
        }
    }
    content.extend(provider_context_content_blocks(
        provider,
        message.provider_context.as_deref(),
        is_hosted_tool_result_block,
    ));
    Ok(content)
}

fn provider_context_content_blocks(
    provider: &ProviderId,
    provider_context: Option<&[ProviderContextBlock]>,
    predicate: fn(&Map<String, Value>) -> bool,
) -> Vec<Value> {
    let Some(provider_context) = provider_context else {
        return Vec::new();
    };

    provider_context
        .iter()
        .filter(|block| {
            block.provider == provider.as_str() && block.kind == "anthropic.content_block"
        })
        .filter_map(|block| block.value.as_object())
        .filter(|block| predicate(block))
        .cloned()
        .map(Value::Object)
        .collect()
}

fn tool_call_to_anthropic(
    provider: &ProviderId,
    tool_call: &ToolCall,
    tool_name_forward_map: &HashMap<String, String>,
) -> SigmaResult<Option<Value>> {
    match tool_call {
        ToolCall::Function(tool_call) => {
            let name = tool_name_forward_map
                .get(&tool_call.function.name)
                .cloned()
                .unwrap_or_else(|| tool_call.function.name.clone());
            let input =
                serde_json::from_str::<Value>(&tool_call.function.arguments).map_err(|err| {
                    SigmaError::ProviderTransform {
                        provider: provider.clone(),
                        message: format!("tool call arguments must be valid JSON: {err}"),
                    }
                })?;
            Ok(Some(json!({
                "type": "tool_use",
                "id": tool_call.id,
                "name": name,
                "input": input
            })))
        }
        ToolCall::Custom(_) => Ok(None),
    }
}

fn tool_result_block(message: &ToolMessage) -> Value {
    let content = match &message.content {
        ToolContent::Text(text) => Value::String(text.clone()),
        ToolContent::Parts(parts) => Value::Array(
            parts
                .iter()
                .map(|part| text_content_block(&part.text, part.cache_control.as_ref()))
                .collect(),
        ),
    };
    json!({
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": content
    })
}

fn image_content_block(provider: &ProviderId, part: &ImagePart) -> SigmaResult<Value> {
    source_from_url(provider, &part.image.url).map(|source| {
        let mut block = Map::new();
        block.insert("type".to_string(), Value::String("image".to_string()));
        block.insert("source".to_string(), source);
        insert_cache_control(&mut block, part.cache_control.as_ref());
        Value::Object(block)
    })
}

fn file_content_block(provider: &ProviderId, part: &FilePart) -> SigmaResult<Value> {
    if let Some(file_id) = &part.file.id {
        let mut document = Map::new();
        document.insert("type".to_string(), Value::String("document".to_string()));
        document.insert(
            "source".to_string(),
            json!({"type": "file", "file_id": file_id}),
        );
        insert_cache_control(&mut document, part.cache_control.as_ref());
        return Ok(Value::Object(document));
    }
    if let Some(file_data) = &part.file.data {
        let mut document = Map::new();
        document.insert("type".to_string(), Value::String("document".to_string()));
        document.insert("source".to_string(), source_from_url(provider, file_data)?);
        if let Some(filename) = &part.file.filename {
            document.insert("title".to_string(), Value::String(filename.clone()));
        }
        insert_cache_control(&mut document, part.cache_control.as_ref());
        return Ok(Value::Object(document));
    }
    Err(SigmaError::ProviderTransform {
        provider: provider.clone(),
        message: "file message part requires id or data".to_string(),
    })
}

fn source_from_url(provider: &ProviderId, url: &str) -> SigmaResult<Value> {
    if let Some((media_type, data)) = parse_data_uri(url) {
        Ok(json!({
            "type": "base64",
            "media_type": media_type,
            "data": data
        }))
    } else if url.starts_with("http://") || url.starts_with("https://") {
        Ok(json!({"type": "url", "url": url}))
    } else {
        Err(SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: "anthropic image/document URL must be http(s) or data URI".to_string(),
        })
    }
}

fn parse_data_uri(value: &str) -> Option<(&str, &str)> {
    let value = value.strip_prefix("data:")?;
    let (media_type, data) = value.split_once(";base64,")?;
    Some((media_type, data))
}

fn non_empty_text(text: &str) -> &str {
    if text.is_empty() { " " } else { text }
}

fn text_content_block(text: &str, cache_control: Option<&CacheControl>) -> Value {
    let mut block = Map::new();
    block.insert("type".to_string(), Value::String("text".to_string()));
    block.insert("text".to_string(), Value::String(text.to_string()));
    insert_cache_control(&mut block, cache_control);
    Value::Object(block)
}

fn insert_cache_control(block: &mut Map<String, Value>, cache_control: Option<&CacheControl>) {
    if let Some(cache_control) = cache_control {
        block.insert(
            "cache_control".to_string(),
            cache_control_to_value(cache_control),
        );
    }
}

fn cache_control_to_value(cache_control: &CacheControl) -> Value {
    let mut value = Map::new();
    let type_value = match cache_control.r#type {
        CacheControlType::Ephemeral => "ephemeral",
    };
    value.insert("type".to_string(), Value::String(type_value.to_string()));
    if let Some(ttl) = cache_control.ttl {
        let ttl_value = match ttl {
            CacheControlTtl::FiveMinutes => "5m",
            CacheControlTtl::OneHour => "1h",
        };
        value.insert("ttl".to_string(), Value::String(ttl_value.to_string()));
    }
    Value::Object(value)
}

fn reasoning_block_to_anthropic(block: &ReasoningBlock) -> Option<Value> {
    match block {
        ReasoningBlock::Text { text, signature } => {
            let mut value = Map::new();
            value.insert("type".to_string(), Value::String("thinking".to_string()));
            value.insert("thinking".to_string(), Value::String(text.clone()));
            if let Some(signature) = signature {
                value.insert("signature".to_string(), Value::String(signature.clone()));
            }
            Some(Value::Object(value))
        }
        ReasoningBlock::Redacted { data, signature } => {
            let mut value = Map::new();
            value.insert(
                "type".to_string(),
                Value::String("redacted_thinking".to_string()),
            );
            value.insert("data".to_string(), Value::String(data.clone()));
            if let Some(signature) = signature {
                value.insert("signature".to_string(), Value::String(signature.clone()));
            }
            Some(Value::Object(value))
        }
        ReasoningBlock::Signature { .. } => None,
    }
}

fn append_anthropic_message(output: &mut Vec<Value>, role: &str, content: Vec<Value>) {
    if let Some(last) = output.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
        && let Some(last_content) = last.get_mut("content").and_then(Value::as_array_mut)
    {
        last_content.extend(content);
        return;
    }
    output.push(json!({ "role": role, "content": content }));
}

fn trim_final_assistant_text(output: &mut [Value]) {
    let Some(last) = output.last_mut() else {
        return;
    };
    if last.get("role").and_then(Value::as_str) != Some("assistant") {
        return;
    }
    let Some(content) = last.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    for block in content {
        if block.get("type").and_then(Value::as_str) == Some("text") {
            let Some(text) = block.get("text").and_then(Value::as_str) else {
                continue;
            };
            let trimmed = text.trim_end().to_string();
            if let Some(object) = block.as_object_mut() {
                object.insert("text".to_string(), Value::String(trimmed));
            }
        }
    }
}

fn infer_beta_headers(params: &ChatParameterMap, beta_values: &mut BTreeSet<String>) {
    if params.get("context_management").is_some() {
        beta_values.insert("context-management-2025-06-27".to_string());
        if context_management_has_compact(params.get("context_management")) {
            beta_values.insert("compact-2026-01-12".to_string());
        }
    }
    if params.get("output_format").is_some() {
        beta_values.insert("structured-outputs-2025-11-13".to_string());
    }
    if params.get("speed").and_then(Value::as_str) == Some("fast") {
        beta_values.insert("fast-mode-2026-02-01".to_string());
    }
    if params.get("mcp_servers").is_some() {
        beta_values.insert("mcp-client-2025-04-04".to_string());
    }
    if params
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| tools.iter().any(is_web_search_tool))
    {
        beta_values.insert("web-search-2025-03-05".to_string());
    }
}

fn context_management_has_compact(value: Option<&Value>) -> bool {
    let edits = value
        .and_then(|value| value.get("edits").or(Some(value)))
        .and_then(Value::as_array);
    edits.is_some_and(|edits| {
        edits.iter().any(|edit| {
            edit.get("type")
                .and_then(Value::as_str)
                .is_some_and(|value| value.contains("compact"))
        })
    })
}

fn is_web_search_tool(tool: &Value) -> bool {
    tool.get("type")
        .and_then(Value::as_str)
        .is_some_and(|value| value.starts_with("web_search"))
}

fn is_compaction_block(block: &Map<String, Value>) -> bool {
    block.get("type").and_then(Value::as_str) == Some("compaction")
}

fn is_hosted_tool_result_block(block: &Map<String, Value>) -> bool {
    block
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|value| value.ends_with("_tool_result"))
}

fn provider_context_block(provider: &ProviderId, kind: &str, value: Value) -> ProviderContextBlock {
    ProviderContextBlock::new(provider.to_string(), kind, value)
}

fn anthropic_response_to_chat(
    context: &ChatAdapterContext<'_>,
    body: Value,
) -> SigmaResult<ChatResponse> {
    if body.get("error").is_some() {
        return Err(error_from_body(context, StatusCode::OK, body));
    }
    let content = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| SigmaError::ProviderResponse {
            provider: context.provider.to_owned(),
            message: "anthropic response missing content array".to_string(),
        })?;
    let reverse_map = reverse_tool_map(context);
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    let mut reasoning_blocks = Vec::new();
    let mut reasoning_content = String::new();
    let mut annotations = Vec::new();
    let mut provider_context = Vec::new();

    for (index, block) in content.iter().enumerate() {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(value) = block.get("text").and_then(Value::as_str) {
                    text.push_str(value);
                }
                collect_citations(block, &mut annotations);
            }
            Some("tool_use" | "server_tool_use") => {
                tool_calls.push(tool_use_to_openai(block, index as u32, &reverse_map));
            }
            Some("thinking") => {
                let block = reasoning_response_block(block);
                if let Some(thinking) = block.text_value() {
                    reasoning_content.push_str(thinking);
                }
                reasoning_blocks.push(block);
            }
            Some("redacted_thinking") => {
                reasoning_blocks.push(reasoning_response_block(block));
            }
            Some(value) if value.ends_with("_tool_result") => {
                provider_context.push(provider_context_block(
                    context.provider,
                    "anthropic.content_block",
                    block.clone(),
                ));
            }
            Some("compaction") => {
                provider_context.push(provider_context_block(
                    context.provider,
                    "anthropic.content_block",
                    block.clone(),
                ));
            }
            _ => {}
        }
    }
    if let Some(context_management) = body.get("context_management") {
        provider_context.push(provider_context_block(
            context.provider,
            "anthropic.response_field",
            json!({
                "name": "context_management",
                "value": context_management,
            }),
        ));
    }
    if let Some(container) = body.get("container") {
        provider_context.push(provider_context_block(
            context.provider,
            "anthropic.response_field",
            json!({
                "name": "container",
                "value": container,
            }),
        ));
    }

    let usage = body.get("usage").and_then(Value::as_object).map(|usage| {
        anthropic_usage(
            usage,
            if reasoning_content.is_empty() {
                None
            } else {
                Some(&reasoning_content)
            },
        )
    });
    let finish_reason = body
        .get("stop_reason")
        .and_then(Value::as_str)
        .map(map_finish_reason);

    Ok(ChatResponse {
        id: body
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("msg_anthropic")
            .to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                content: if text.is_empty() { None } else { Some(text) },
                refusal: None,
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
                annotations: if annotations.is_empty() {
                    None
                } else {
                    Some(annotations)
                },
                role: Role::Assistant,
                audio: None,
                reasoning: if reasoning_blocks.is_empty() {
                    None
                } else {
                    Some(reasoning_blocks)
                },
                provider_context: if provider_context.is_empty() {
                    None
                } else {
                    Some(provider_context)
                },
            },
            finish_reason,
            logprobs: None,
        }],
        created: current_unix_timestamp(),
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_else(|| context.provider_model.as_str())
            .to_string(),
        service_tier: None,
        object: "chat.completion".to_string(),
        usage,
    })
}

fn reasoning_response_block(block: &Value) -> ReasoningBlock {
    match block.get("type").and_then(Value::as_str) {
        Some("redacted_thinking") => ReasoningBlock::redacted(
            block
                .get("data")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            block.get("signature").and_then(Value::as_str),
        ),
        _ => ReasoningBlock::text(
            block
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            block.get("signature").and_then(Value::as_str),
        ),
    }
}

fn collect_citations(block: &Value, annotations: &mut Vec<Annotation>) {
    let Some(citations) = block.get("citations").and_then(Value::as_array) else {
        return;
    };
    for citation in citations {
        let title = citation
            .get("document_title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let start = citation
            .get("start_char_index")
            .or_else(|| citation.get("start_page_number"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let end = citation
            .get("end_char_index")
            .or_else(|| citation.get("end_page_number"))
            .and_then(Value::as_u64)
            .unwrap_or(start as u64) as u32;
        annotations.push(Annotation::UrlCitation {
            url_citation: UrlCitation {
                start_index: start,
                end_index: end,
                title,
                url: citation
                    .get("url")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            },
        });
    }
}

fn tool_use_to_openai(
    block: &Value,
    index: u32,
    reverse_map: &HashMap<String, String>,
) -> ToolCall {
    let name = block
        .get("name")
        .and_then(Value::as_str)
        .map(|name| {
            reverse_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_string())
        })
        .unwrap_or_default();
    let input = block
        .get("input")
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    let arguments = serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());

    let _ = index;
    ToolCall::Function(FunctionToolCall {
        id: block
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        function: FunctionCall { name, arguments },
        reasoning: None,
    })
}

fn anthropic_usage(usage: &Map<String, Value>, reasoning_content: Option<&str>) -> Usage {
    let raw_input = u32_field(usage, "input_tokens");
    let cache_creation = u32_field(usage, "cache_creation_input_tokens");
    let cache_read = u32_field(usage, "cache_read_input_tokens");
    let prompt_tokens = raw_input + cache_creation + cache_read;
    let completion_tokens = u32_field(usage, "output_tokens");
    let reasoning_tokens = reasoning_content
        .filter(|value| !value.is_empty())
        .map(|_| 0);
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        cache_creation_input_tokens: optional_nonzero(cache_creation),
        cache_read_input_tokens: optional_nonzero(cache_read),
        hosted_tool_use: usage.get("server_tool_use").and_then(server_tool_use),
        inference_geo: usage
            .get("inference_geo")
            .and_then(Value::as_str)
            .map(str::to_string),
        speed: usage
            .get("speed")
            .and_then(Value::as_str)
            .map(str::to_string),
        prompt_tokens_details: Some(PromptTokensDetails {
            audio_tokens: None,
            cached_tokens: optional_nonzero(cache_read),
            text_tokens: None,
            image_tokens: None,
            video_tokens: None,
        }),
        completion_tokens_details: Some(CompletionTokensDetails {
            accepted_prediction_tokens: None,
            audio_tokens: None,
            text_tokens: None,
            image_tokens: None,
            video_tokens: None,
            reasoning_tokens,
            rejected_prediction_tokens: None,
        }),
    }
}

fn server_tool_use(value: &Value) -> Option<HostedToolUsage> {
    let object = value.as_object()?;
    Some(HostedToolUsage {
        web_search_requests: optional_nonzero(u32_field(object, "web_search_requests")),
        tool_search_requests: optional_nonzero(u32_field(object, "tool_search_requests")),
    })
}

fn optional_nonzero(value: u32) -> Option<u32> {
    if value == 0 { None } else { Some(value) }
}

fn u32_field(object: &Map<String, Value>, key: &str) -> u32 {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0)
}

fn map_finish_reason(value: &str) -> FinishReason {
    match value {
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "stop_sequence" | "end_turn" => FinishReason::Stop,
        _ => FinishReason::Stop,
    }
}

struct AnthropicSseStream {
    provider: ProviderId,
    stream: ProviderByteStream,
    buffer: String,
    pending: std::collections::VecDeque<SigmaResult<ChatStreamChunk>>,
    done: bool,
    id: String,
    model: String,
    created: u32,
    prompt_tokens: u32,
    current_tool_index: Option<u32>,
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
    reverse_tool_map: HashMap<String, String>,
}

impl AnthropicSseStream {
    fn new(
        provider: ProviderId,
        stream: ProviderByteStream,
        reverse_tool_map: HashMap<String, String>,
    ) -> Self {
        Self {
            provider,
            stream,
            buffer: String::new(),
            pending: std::collections::VecDeque::new(),
            done: false,
            id: "msg_anthropic_stream".to_string(),
            model: String::new(),
            created: current_unix_timestamp(),
            prompt_tokens: 0,
            current_tool_index: None,
            current_tool_id: None,
            current_tool_name: None,
            reverse_tool_map,
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
        if flush {
            let event = self.buffer.trim().to_string();
            self.buffer.clear();
            if !event.is_empty() {
                self.push_event(&event);
            }
        }
    }

    fn push_event(&mut self, event: &str) {
        let Some(data) = event_data(event) else {
            return;
        };
        let value = match serde_json::from_str::<Value>(&data) {
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
        self.handle_stream_value(value);
    }

    fn handle_stream_value(&mut self, value: Value) {
        match value.get("type").and_then(Value::as_str) {
            Some("message_start") => self.handle_message_start(&value),
            Some("content_block_start") => self.handle_content_block_start(&value),
            Some("content_block_delta") => self.handle_content_block_delta(&value),
            Some("content_block_stop") => {
                self.current_tool_index = None;
                self.current_tool_id = None;
                self.current_tool_name = None;
            }
            Some("message_delta") => self.handle_message_delta(&value),
            Some("message_stop") => self.done = true,
            Some("ping") => {}
            Some("error") => {
                let error = value.get("error").cloned().unwrap_or(value);
                let message = error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("anthropic stream error")
                    .to_string();
                let code = error
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                self.done = true;
                self.pending.push_back(Err(SigmaError::ProviderBusiness {
                    provider: self.provider.clone(),
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    code,
                    message,
                    details: Some(error),
                }));
            }
            _ => {}
        }
    }

    fn handle_message_start(&mut self, value: &Value) {
        let Some(message) = value.get("message").and_then(Value::as_object) else {
            return;
        };
        if let Some(id) = message.get("id").and_then(Value::as_str) {
            self.id = id.to_string();
        }
        if let Some(model) = message.get("model").and_then(Value::as_str) {
            self.model = model.to_string();
        }
        if let Some(usage) = message.get("usage").and_then(Value::as_object) {
            self.prompt_tokens = u32_field(usage, "input_tokens")
                + u32_field(usage, "cache_creation_input_tokens")
                + u32_field(usage, "cache_read_input_tokens");
        }
    }

    fn handle_content_block_start(&mut self, value: &Value) {
        let Some(block) = value.get("content_block") else {
            return;
        };
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    self.pending.push_back(Ok(self.chunk(
                        Some(text.to_string()),
                        None,
                        None,
                        None,
                        None,
                    )));
                }
            }
            Some("tool_use" | "server_tool_use") => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or(0);
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|name| {
                        self.reverse_tool_map
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| name.to_string())
                    })
                    .unwrap_or_default();
                self.current_tool_index = Some(index);
                self.current_tool_id = Some(id.clone());
                self.current_tool_name = Some(name.clone());
                self.pending.push_back(Ok(self.chunk(
                    None,
                    Some(vec![ToolCallDelta {
                        index,
                        id: Some(id),
                        r#type: Some(ToolCallKind::Function),
                        function: Some(FunctionCallDelta {
                            name: Some(name),
                            arguments: Some(String::new()),
                        }),
                        reasoning: None,
                    }]),
                    None,
                    None,
                    None,
                )));
            }
            Some("thinking" | "redacted_thinking") => {
                let thinking = reasoning_response_block(block);
                self.pending.push_back(Ok(self.chunk(
                    None,
                    None,
                    None,
                    Some(vec![thinking]),
                    None,
                )));
            }
            _ => {}
        }
    }

    fn handle_content_block_delta(&mut self, value: &Value) {
        let Some(delta) = value.get("delta") else {
            return;
        };
        match delta.get("type").and_then(Value::as_str) {
            Some("text_delta") => {
                if let Some(text) = delta.get("text").and_then(Value::as_str) {
                    self.pending.push_back(Ok(self.chunk(
                        Some(text.to_string()),
                        None,
                        None,
                        None,
                        None,
                    )));
                }
            }
            Some("input_json_delta") => {
                let arguments = delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.pending.push_back(Ok(self.chunk(
                    None,
                    Some(vec![ToolCallDelta {
                        index: self.current_tool_index.unwrap_or_else(|| {
                            value
                                .get("index")
                                .and_then(Value::as_u64)
                                .and_then(|value| u32::try_from(value).ok())
                                .unwrap_or(0)
                        }),
                        id: None,
                        r#type: None,
                        function: Some(FunctionCallDelta {
                            name: None,
                            arguments: Some(arguments),
                        }),
                        reasoning: None,
                    }]),
                    None,
                    None,
                    None,
                )));
            }
            Some("thinking_delta" | "signature_delta") => {
                let thinking = ReasoningBlock::text(
                    delta
                        .get("thinking")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    delta.get("signature").and_then(Value::as_str),
                );
                self.pending.push_back(Ok(self.chunk(
                    None,
                    None,
                    None,
                    Some(vec![thinking]),
                    None,
                )));
            }
            _ => {}
        }
    }

    fn handle_message_delta(&mut self, value: &Value) {
        let finish_reason = value
            .get("delta")
            .and_then(|delta| delta.get("stop_reason"))
            .and_then(Value::as_str)
            .map(map_finish_reason);
        let usage = value.get("usage").and_then(Value::as_object).map(|usage| {
            let completion_tokens = u32_field(usage, "output_tokens");
            Usage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens,
                total_tokens: self.prompt_tokens + completion_tokens,
                cache_creation_input_tokens: optional_nonzero(u32_field(
                    usage,
                    "cache_creation_input_tokens",
                )),
                cache_read_input_tokens: optional_nonzero(u32_field(
                    usage,
                    "cache_read_input_tokens",
                )),
                hosted_tool_use: usage.get("server_tool_use").and_then(server_tool_use),
                inference_geo: usage
                    .get("inference_geo")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                speed: usage
                    .get("speed")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                prompt_tokens_details: Some(PromptTokensDetails {
                    audio_tokens: None,
                    cached_tokens: optional_nonzero(u32_field(usage, "cache_read_input_tokens")),
                    text_tokens: None,
                    image_tokens: None,
                    video_tokens: None,
                }),
                completion_tokens_details: Some(CompletionTokensDetails {
                    accepted_prediction_tokens: None,
                    audio_tokens: None,
                    text_tokens: None,
                    image_tokens: None,
                    video_tokens: None,
                    reasoning_tokens: None,
                    rejected_prediction_tokens: None,
                }),
            }
        });
        self.pending
            .push_back(Ok(self.chunk(None, None, finish_reason, None, usage)));
    }

    fn chunk(
        &self,
        content: Option<String>,
        tool_calls: Option<Vec<ToolCallDelta>>,
        finish_reason: Option<FinishReason>,
        reasoning: Option<Vec<ReasoningBlock>>,
        usage: Option<Usage>,
    ) -> ChatStreamChunk {
        ChatStreamChunk {
            id: self.id.clone(),
            choices: vec![ChatStreamChoice {
                index: 0,
                delta: AssistantDelta {
                    content,
                    tool_calls,
                    role: None,
                    refusal: None,
                    reasoning,
                },
                finish_reason,
                logprobs: None,
            }],
            created: self.created,
            model: self.model.clone(),
            service_tier: None,
            object: "chat.completion.chunk".to_string(),
            usage,
        }
    }
}

impl Stream for AnthropicSseStream {
    type Item = SigmaResult<ChatStreamChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(item) = self.pending.pop_front() {
            return Poll::Ready(Some(item));
        }
        if self.done {
            return Poll::Ready(None);
        }

        loop {
            match self.stream.as_mut().poll_next(cx) {
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

fn event_data(event: &str) -> Option<String> {
    let data_lines = event
        .lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .map(str::trim_start)
        .collect::<Vec<_>>();
    if data_lines.is_empty() {
        None
    } else {
        Some(data_lines.join("\n"))
    }
}

fn reverse_tool_map(context: &ChatAdapterContext<'_>) -> HashMap<String, String> {
    context
        .provider_state
        .as_ref()
        .and_then(|state| state.get(TOOL_NAME_MAP_STATE_KEY))
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .filter_map(|(key, value)| {
                    value.as_str().map(|value| (key.clone(), value.to_string()))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_response_json(provider: &ProviderId, body: &[u8]) -> SigmaResult<Value> {
    serde_json::from_slice(body).map_err(|err| SigmaError::ProviderResponse {
        provider: provider.clone(),
        message: err.to_string(),
    })
}

fn anthropic_error_response(
    context: &ChatAdapterContext<'_>,
    response: ProviderResponse,
) -> SigmaError {
    let body = serde_json::from_slice::<Value>(&response.body).ok();
    match body {
        Some(body) => error_from_body(context, response.status, body),
        None => SigmaError::ProviderBusiness {
            provider: context.provider.to_owned(),
            status: response.status,
            code: None,
            message: fallback_error_message(response.status, &response.body),
            details: None,
        },
    }
}

fn error_from_body(
    context: &ChatAdapterContext<'_>,
    status: StatusCode,
    body: Value,
) -> SigmaError {
    let error = body.get("error").filter(|error| error.is_object());
    let code = error
        .and_then(|error| error.get("type").or_else(|| error.get("code")))
        .and_then(Value::as_str)
        .map(str::to_string);
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| body.to_string());
    SigmaError::ProviderBusiness {
        provider: context.provider.to_owned(),
        status,
        code,
        message,
        details: error.cloned().or(Some(body)),
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

fn resolve_api_base(init: &ProviderInit<AnthropicConfig>) -> String {
    init.common
        .api_base
        .clone()
        .or_else(|| non_empty_env("ANTHROPIC_API_BASE"))
        .or_else(|| non_empty_env("ANTHROPIC_BASE_URL"))
        .unwrap_or_else(|| ANTHROPIC_DEFAULT_BASE_URL.to_string())
}

fn resolve_api_key(api_key: Option<SecretString>) -> Option<SecretString> {
    api_key.or_else(|| non_empty_env("ANTHROPIC_API_KEY").map(SecretString::from))
}

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

fn messages_url(api_base: &str) -> String {
    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with("/v1/messages") {
        api_base.to_string()
    } else if api_base.ends_with("/v1") {
        format!("{api_base}/messages")
    } else {
        format!("{api_base}/v1/messages")
    }
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

fn insert_header_if_missing(
    provider: &ProviderId,
    headers: &mut HeaderMap,
    name: HeaderName,
    value: &str,
) -> SigmaResult<()> {
    if !headers.contains_key(&name) {
        headers.insert(name.clone(), header_value(provider, name.as_str(), value)?);
    }
    Ok(())
}

fn header_value(provider: &ProviderId, name: &str, value: &str) -> SigmaResult<HeaderValue> {
    HeaderValue::from_str(value).map_err(|err| SigmaError::ProviderSigning {
        provider: provider.clone(),
        message: format!("invalid header value for `{name}`: {err}"),
    })
}

fn merge_header_beta_values(headers: &HeaderMap, beta_values: &mut BTreeSet<String>) {
    if let Some(value) = headers
        .get("anthropic-beta")
        .and_then(|value| value.to_str().ok())
    {
        insert_split_beta(beta_values, value);
    }
}

fn add_beta_header(provider: &ProviderId, headers: &mut HeaderMap, beta: &str) -> SigmaResult<()> {
    let mut values = BTreeSet::new();
    merge_header_beta_values(headers, &mut values);
    insert_split_beta(&mut values, beta);
    let value = values.into_iter().collect::<Vec<_>>().join(",");
    headers.insert(
        HeaderName::from_static("anthropic-beta"),
        header_value(provider, "anthropic-beta", &value)?,
    );
    Ok(())
}

fn insert_split_beta(values: &mut BTreeSet<String>, beta: &str) {
    for value in beta
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        values.insert(value.to_string());
    }
}

fn current_unix_timestamp() -> u32 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u32::try_from(duration.as_secs()).ok())
        .unwrap_or(u32::MAX)
}

submit_provider! {
    kind: ANTHROPIC_KIND,
    constructor: AnthropicProvider::from_config,
    config: AnthropicConfig,
}
