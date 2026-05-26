use std::collections::{BTreeSet, HashMap};

use http::{HeaderMap, HeaderName};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::config::ChatParameterMap;
use crate::providers::common::{parse_data_uri, signing_header_value as header_value};
use crate::types::chat::{
    AssistantContent, AssistantContentPart, AssistantMessage, CacheControl, CacheControlTtl,
    CacheControlType, ChatMessage, FilePart, ImagePart, ProviderContextBlock, ReasoningBlock,
    TextContent, ToolCall, ToolContent, ToolMessage, UserContent, UserContentPart,
};
use crate::types::shared::ResponseFormat;
use crate::{ModelName, ProviderId, SigmaError, SigmaResult};

use super::{AnthropicChatAdapter, RESPONSE_FORMAT_TOOL_NAME};

impl AnthropicChatAdapter {
    pub(super) fn collect_beta_values(
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

pub(super) struct TranslatedMessages {
    pub(super) messages: Vec<Value>,
    pub(super) system: Vec<Value>,
}

#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
struct AnthropicThinkingParam {
    #[serde(rename = "type")]
    r#type: AnthropicThinkingType,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum AnthropicThinkingType {
    Enabled,
    Adaptive,
}

pub(super) fn map_token_params(params: &mut ChatParameterMap, default_max_tokens: u32) {
    if let Some(value) = params.remove("max_completion_tokens") {
        params.entry("max_tokens".to_string()).or_insert(value);
    }
    params
        .entry("max_tokens".to_string())
        .or_insert_with(|| Value::from(default_max_tokens));
}

pub(super) fn map_stop_sequences(params: &mut ChatParameterMap) {
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

pub(super) fn map_reasoning_effort(
    params: &mut ChatParameterMap,
    model: &ModelName,
) -> SigmaResult<()> {
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

pub(super) fn map_user_metadata(params: &mut ChatParameterMap) {
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

pub(super) fn map_response_format(
    params: &mut ChatParameterMap,
    model: &ModelName,
) -> SigmaResult<()> {
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

pub(super) fn provider_options_contain(
    provider_options: Option<&ChatParameterMap>,
    key: &str,
) -> bool {
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

pub(super) fn map_tool_choice(params: &mut ChatParameterMap) {
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

pub(super) fn map_web_search_tool(params: &mut ChatParameterMap) {
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

pub(super) fn filter_metadata(params: &mut ChatParameterMap) {
    let Some(metadata) = params.get_mut("metadata").and_then(Value::as_object_mut) else {
        return;
    };
    let user_id = metadata.get("user_id").cloned();
    metadata.clear();
    if let Some(user_id) = user_id {
        metadata.insert("user_id".to_string(), user_id);
    }
}

pub(super) fn is_internal_param(key: &str) -> bool {
    matches!(key, "anthropic_beta" | "reasoning_effort")
}

pub(super) fn translate_anthropic_messages(
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

pub(super) fn infer_beta_headers(params: &ChatParameterMap, beta_values: &mut BTreeSet<String>) {
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

pub(super) fn provider_context_block(
    provider: &ProviderId,
    kind: &str,
    value: Value,
) -> ProviderContextBlock {
    ProviderContextBlock::new(provider.to_string(), kind, value)
}

pub(super) fn messages_url(api_base: &str) -> String {
    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with("/v1/messages") {
        api_base.to_string()
    } else if api_base.ends_with("/v1") {
        format!("{api_base}/messages")
    } else {
        format!("{api_base}/v1/messages")
    }
}

pub(super) fn insert_header_if_missing(
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

pub(super) fn merge_header_beta_values(headers: &HeaderMap, beta_values: &mut BTreeSet<String>) {
    if let Some(value) = headers
        .get("anthropic-beta")
        .and_then(|value| value.to_str().ok())
    {
        insert_split_beta(beta_values, value);
    }
}

pub(super) fn add_beta_header(
    provider: &ProviderId,
    headers: &mut HeaderMap,
    beta: &str,
) -> SigmaResult<()> {
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
