use std::collections::HashMap;

use http::Method;
use serde_json::{Map, Value, json};

use crate::config::ChatParameterMap;
use crate::providers::common::parse_data_uri;
use crate::types::chat::{
    AssistantContent, AssistantContentPart, CacheControl, CacheControlTtl, ChatMessage,
    FunctionToolCall, ReasoningBlock, TextContent, TextPart, ToolCall, ToolChoice, ToolChoiceMode,
    ToolContent, ToolDefinition, UserContent, UserContentPart,
};
use crate::types::shared::{FunctionObject, ReasoningEffort, ResponseFormat};
use crate::{ChatAdapterContext, ModelName, ProviderEndpoint, ProviderId, SigmaError, SigmaResult};

use super::{JSON_TOOL_NAME, TOOL_NAME_MAP_STATE_KEY};

pub(super) const SIGNING_REGION_STATE_KEY: &str = "bedrock_signing_region";

pub(super) struct BedrockRequestBody {
    pub(super) body: Map<String, Value>,
    pub(super) reverse_tool_map: HashMap<String, String>,
}

pub(super) fn converse_url(api_base: &str, model: &ModelName, stream: bool) -> String {
    let endpoint = if stream {
        "converse-stream"
    } else {
        "converse"
    };
    format!(
        "{}/model/{}/{}",
        api_base.trim_end_matches('/'),
        encode_model_id(model.as_str()),
        endpoint
    )
}

pub(super) fn endpoint(api_base: &str, model: &ModelName, stream: bool) -> ProviderEndpoint {
    ProviderEndpoint {
        method: Method::POST,
        url: converse_url(api_base, model, stream),
    }
}

pub(super) fn bedrock_request_body(
    provider: &ProviderId,
    context: ChatAdapterContext<'_>,
    messages: &[ChatMessage],
    params: &ChatParameterMap,
) -> SigmaResult<BedrockRequestBody> {
    let mut params = params.clone();
    params.remove("stream");
    let translated = translate_messages(provider, messages)?;
    let mapped = map_params(provider, context.provider_model, &mut params)?;
    let mut body = Map::new();

    body.insert("messages".to_string(), Value::Array(translated.messages));
    if !translated.system.is_empty() {
        body.insert("system".to_string(), Value::Array(translated.system));
    }
    if !mapped.inference_config.is_empty() {
        body.insert(
            "inferenceConfig".to_string(),
            Value::Object(mapped.inference_config),
        );
    }
    if !mapped.additional_model_request_fields.is_empty() {
        body.insert(
            "additionalModelRequestFields".to_string(),
            Value::Object(mapped.additional_model_request_fields),
        );
    }
    if let Some(tool_config) = mapped.tool_config {
        body.insert("toolConfig".to_string(), tool_config);
    }
    if let Some(request_metadata) = mapped.request_metadata {
        body.insert("requestMetadata".to_string(), request_metadata);
    }
    if let Some(output_config) = mapped.output_config {
        body.insert("outputConfig".to_string(), output_config);
    }
    if let Some(guardrail_config) = mapped.guardrail_config {
        body.insert("guardrailConfig".to_string(), guardrail_config);
    }
    if let Some(performance_config) = mapped.performance_config {
        body.insert("performanceConfig".to_string(), performance_config);
    }
    if let Some(service_tier) = mapped.service_tier {
        body.insert("serviceTier".to_string(), service_tier);
    }

    Ok(BedrockRequestBody {
        body,
        reverse_tool_map: mapped.reverse_tool_map,
    })
}

struct TranslatedMessages {
    messages: Vec<Value>,
    system: Vec<Value>,
}

fn translate_messages(
    provider: &ProviderId,
    messages: &[ChatMessage],
) -> SigmaResult<TranslatedMessages> {
    let mut system = Vec::new();
    let mut bedrock_messages = Vec::new();

    for message in messages {
        match message {
            ChatMessage::Developer(message) => {
                system.extend(text_content_blocks(&message.content));
            }
            ChatMessage::System(message) => {
                system.extend(text_content_blocks(&message.content));
            }
            ChatMessage::User(message) => {
                let content = user_content_blocks(provider, &message.content)?;
                if !content.is_empty() {
                    push_message(&mut bedrock_messages, "user", content);
                }
            }
            ChatMessage::Assistant(message) => {
                let mut content = Vec::new();
                if let Some(reasoning) = &message.reasoning {
                    content.extend(reasoning_blocks(reasoning));
                }
                if let Some(assistant_content) = &message.content {
                    content.extend(assistant_content_blocks(assistant_content));
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        content.push(tool_use_block(tool_call)?);
                    }
                }
                if !content.is_empty() {
                    push_message(&mut bedrock_messages, "assistant", content);
                }
            }
            ChatMessage::Tool(message) => {
                let content = vec![tool_result_block(&message.tool_call_id, &message.content)];
                push_message(&mut bedrock_messages, "user", content);
            }
        }
    }

    if bedrock_messages.is_empty() {
        bedrock_messages.push(json!({
            "role": "user",
            "content": [{"text": " "}]
        }));
    }

    Ok(TranslatedMessages {
        messages: bedrock_messages,
        system,
    })
}

fn push_message(messages: &mut Vec<Value>, role: &str, content: Vec<Value>) {
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some(role)
        && let Some(existing) = last.get_mut("content").and_then(Value::as_array_mut)
    {
        existing.extend(content);
        return;
    }

    messages.push(json!({
        "role": role,
        "content": content,
    }));
}

fn text_content_blocks(content: &TextContent) -> Vec<Value> {
    match content {
        TextContent::Text(text) => nonempty_text_block(text, None).into_iter().collect(),
        TextContent::Parts(parts) => parts.iter().filter_map(text_part_block).collect(),
    }
}

fn user_content_blocks(provider: &ProviderId, content: &UserContent) -> SigmaResult<Vec<Value>> {
    match content {
        UserContent::Text(text) => Ok(nonempty_text_block(text, None).into_iter().collect()),
        UserContent::Parts(parts) => parts
            .iter()
            .map(|part| user_content_block(provider, part))
            .collect(),
    }
}

fn user_content_block(provider: &ProviderId, part: &UserContentPart) -> SigmaResult<Value> {
    match part {
        UserContentPart::Text(text) => {
            text_part_block(text).ok_or_else(|| SigmaError::ProviderTransform {
                provider: provider.clone(),
                message: "bedrock text content cannot be empty".to_string(),
            })
        }
        UserContentPart::Image(image) => media_block(
            provider,
            "image",
            &image.image.url,
            None,
            image.cache_control.as_ref(),
            None,
        ),
        UserContentPart::File(file) => {
            let source = file
                .file
                .data
                .as_deref()
                .or(file.file.id.as_deref())
                .ok_or_else(|| SigmaError::ProviderTransform {
                    provider: provider.clone(),
                    message: "bedrock file content requires data or id".to_string(),
                })?;
            let media_type = file.file.media_type.as_deref();
            let kind = if media_type.is_some_and(|value| value.starts_with("image/")) {
                "image"
            } else {
                "document"
            };
            media_block(
                provider,
                kind,
                source,
                media_type,
                file.cache_control.as_ref(),
                file.file.filename.as_deref(),
            )
        }
        UserContentPart::Audio(_) => Err(SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: "bedrock converse provider does not support audio input".to_string(),
        }),
    }
}

fn assistant_content_blocks(content: &AssistantContent) -> Vec<Value> {
    match content {
        AssistantContent::Text(text) => nonempty_text_block(text, None).into_iter().collect(),
        AssistantContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                AssistantContentPart::Text(text) => text_part_block(text),
                AssistantContentPart::Refusal(_) => None,
            })
            .collect(),
    }
}

fn text_part_block(part: &TextPart) -> Option<Value> {
    nonempty_text_block(&part.text, part.cache_control.as_ref())
}

fn nonempty_text_block(text: &str, cache_control: Option<&CacheControl>) -> Option<Value> {
    if text.is_empty() {
        return None;
    }
    let mut block = Map::new();
    block.insert("text".to_string(), Value::String(text.to_string()));
    if let Some(cache_control) = cache_control {
        block.insert("cachePoint".to_string(), cache_control_block(cache_control));
    }
    Some(Value::Object(block))
}

fn media_block(
    provider: &ProviderId,
    kind: &str,
    source: &str,
    media_type: Option<&str>,
    cache_control: Option<&CacheControl>,
    name: Option<&str>,
) -> SigmaResult<Value> {
    let (mime_type, data) =
        parse_data_uri(source).ok_or_else(|| SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: "bedrock media content requires a data URI".to_string(),
        })?;
    let format = media_format(media_type.unwrap_or(mime_type)).ok_or_else(|| {
        SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: format!(
                "unsupported bedrock media type `{}`",
                media_type.unwrap_or(mime_type)
            ),
        }
    })?;

    let mut value = if kind == "image" {
        json!({
            "image": {
                "format": format,
                "source": {"bytes": data}
            }
        })
    } else {
        json!({
            "document": {
                "format": format,
                "name": name.unwrap_or("document"),
                "source": {"bytes": data}
            }
        })
    };

    if let Some(cache_control) = cache_control.map(cache_control_block)
        && let Value::Object(block) = &mut value
    {
        block.insert("cachePoint".to_string(), cache_control);
    }

    Ok(value)
}

fn media_format(media_type: &str) -> Option<&str> {
    match media_type {
        "image/jpeg" | "image/jpg" => Some("jpeg"),
        "image/png" => Some("png"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        "application/pdf" => Some("pdf"),
        "text/csv" => Some("csv"),
        "text/html" => Some("html"),
        "text/markdown" => Some("md"),
        "text/plain" => Some("txt"),
        _ => media_type
            .rsplit('/')
            .next()
            .filter(|value| !value.is_empty()),
    }
}

fn tool_result_block(tool_call_id: &str, content: &ToolContent) -> Value {
    let content = match content {
        ToolContent::Text(text) => vec![json!({"text": text})],
        ToolContent::Parts(parts) => parts.iter().filter_map(text_part_block).collect::<Vec<_>>(),
    };
    json!({
        "toolResult": {
            "toolUseId": tool_call_id,
            "content": content,
        }
    })
}

fn tool_use_block(tool_call: &ToolCall) -> SigmaResult<Value> {
    match tool_call {
        ToolCall::Function(tool_call) => Ok(function_tool_use_block(tool_call)),
        ToolCall::Custom(_) => Err(SigmaError::ProviderTransform {
            provider: ProviderId::from("bedrock"),
            message: "bedrock converse provider does not support custom tool replay".to_string(),
        }),
    }
}

fn function_tool_use_block(tool_call: &FunctionToolCall) -> Value {
    json!({
        "toolUse": {
            "toolUseId": tool_call.id,
            "name": tool_call.function.name,
            "input": parse_arguments(&tool_call.function.arguments),
        }
    })
}

fn parse_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

fn reasoning_blocks(reasoning: &[ReasoningBlock]) -> Vec<Value> {
    reasoning
        .iter()
        .map(|block| match block {
            ReasoningBlock::Text { text, signature } => {
                let mut reasoning = json!({"text": text});
                if let Some(signature) = signature
                    && let Value::Object(object) = &mut reasoning
                {
                    object.insert("signature".to_string(), Value::String(signature.clone()));
                }
                json!({"reasoningContent": {"reasoningText": reasoning}})
            }
            ReasoningBlock::Redacted { data, .. } => {
                json!({"reasoningContent": {"redactedContent": data}})
            }
            ReasoningBlock::Signature { value } => {
                json!({"reasoningContent": {"reasoningText": {"signature": value}}})
            }
        })
        .collect()
}

fn cache_control_block(cache_control: &CacheControl) -> Value {
    let mut block = Map::new();
    block.insert("type".to_string(), Value::String("default".to_string()));
    if let Some(ttl) = cache_control.ttl {
        let ttl = match ttl {
            CacheControlTtl::FiveMinutes => "5m",
            CacheControlTtl::OneHour => "1h",
        };
        block.insert("ttl".to_string(), Value::String(ttl.to_string()));
    }
    Value::Object(block)
}

struct MappedParams {
    inference_config: Map<String, Value>,
    additional_model_request_fields: Map<String, Value>,
    tool_config: Option<Value>,
    request_metadata: Option<Value>,
    output_config: Option<Value>,
    guardrail_config: Option<Value>,
    performance_config: Option<Value>,
    service_tier: Option<Value>,
    reverse_tool_map: HashMap<String, String>,
}

fn map_params(
    provider: &ProviderId,
    model: &ModelName,
    params: &mut ChatParameterMap,
) -> SigmaResult<MappedParams> {
    let mut inference_config = Map::new();
    move_param(
        params,
        &mut inference_config,
        "max_completion_tokens",
        "maxTokens",
    );
    move_param(params, &mut inference_config, "max_tokens", "maxTokens");
    move_stop(params, &mut inference_config);
    move_param(params, &mut inference_config, "temperature", "temperature");
    move_param(params, &mut inference_config, "top_p", "topP");

    let mut additional_model_request_fields = Map::new();
    if let Some(top_k) = params.remove("top_k") {
        if is_nova_model(model.as_str()) {
            let mut inference = Map::new();
            inference.insert("topK".to_string(), top_k);
            additional_model_request_fields
                .insert("inferenceConfig".to_string(), Value::Object(inference));
        } else {
            additional_model_request_fields.insert("top_k".to_string(), top_k);
        }
    }
    if let Some(reasoning_effort) = params.remove("reasoning_effort") {
        map_reasoning_effort(
            model,
            reasoning_effort,
            &mut additional_model_request_fields,
        )?;
    }
    if let Some(thinking) = params.remove("thinking") {
        additional_model_request_fields.insert("thinking".to_string(), thinking);
    }
    if let Some(response_format) = params.remove("response_format") {
        add_response_format_tool(provider, response_format, params)?;
    }
    if let Some(web_search) = params.remove("web_search")
        && is_nova_model(model.as_str())
    {
        let mut tools = params
            .remove("tools")
            .and_then(|value| value.as_array().cloned())
            .unwrap_or_default();
        tools.push(json!({"systemTool": {"name": "nova_grounding"}}));
        params.insert("tools".to_string(), Value::Array(tools));
        let _ = web_search;
    }
    params.remove("parallel_tool_calls");
    params.remove("stream_options");

    let (tool_config, reverse_tool_map) = map_tools(
        provider,
        params.remove("tools"),
        params.remove("tool_choice"),
    )?;
    let request_metadata = params.remove("requestMetadata");
    let output_config = params.remove("outputConfig");
    let guardrail_config = params.remove("guardrailConfig");
    let performance_config = params.remove("performanceConfig");
    let service_tier = params.remove("service_tier").map(service_tier_block);

    for (key, value) in std::mem::take(params) {
        additional_model_request_fields.insert(key, value);
    }

    Ok(MappedParams {
        inference_config,
        additional_model_request_fields,
        tool_config,
        request_metadata,
        output_config,
        guardrail_config,
        performance_config,
        service_tier,
        reverse_tool_map,
    })
}

fn move_param(
    params: &mut ChatParameterMap,
    target: &mut Map<String, Value>,
    source: &str,
    dest: &str,
) {
    if let Some(value) = params.remove(source) {
        target.entry(dest.to_string()).or_insert(value);
    }
}

fn move_stop(params: &mut ChatParameterMap, target: &mut Map<String, Value>) {
    let Some(value) = params.remove("stop") else {
        return;
    };
    match value {
        Value::String(value) => {
            target.insert(
                "stopSequences".to_string(),
                Value::Array(vec![Value::String(value)]),
            );
        }
        Value::Array(values) => {
            target.insert("stopSequences".to_string(), Value::Array(values));
        }
        _ => {}
    }
}

fn map_reasoning_effort(
    model: &ModelName,
    value: Value,
    additional: &mut Map<String, Value>,
) -> SigmaResult<()> {
    let effort = serde_json::from_value::<ReasoningEffort>(value).map_err(|err| {
        SigmaError::ProviderTransform {
            provider: ProviderId::from("bedrock"),
            message: err.to_string(),
        }
    })?;
    let effort = serde_json::to_value(effort).map_err(|err| SigmaError::ProviderTransform {
        provider: ProviderId::from("bedrock"),
        message: err.to_string(),
    })?;
    let effort = effort.as_str().unwrap_or("medium");

    if model.as_str().contains("gpt-oss") {
        additional.insert(
            "reasoning_effort".to_string(),
            Value::String(effort.to_string()),
        );
    } else if is_nova_2_model(model.as_str()) {
        additional.insert(
            "reasoningConfig".to_string(),
            json!({"type": "enabled", "maxReasoningEffort": effort}),
        );
    } else if effort != "none" {
        additional.insert(
            "thinking".to_string(),
            json!({"type": "enabled", "budget_tokens": reasoning_budget_tokens(effort)?}),
        );
    }
    Ok(())
}

fn reasoning_budget_tokens(value: &str) -> SigmaResult<u32> {
    match value {
        "minimal" | "low" => Ok(1024),
        "medium" => Ok(2048),
        "high" => Ok(4096),
        "xhigh" => Ok(8192),
        "max" => Ok(16384),
        other => Err(SigmaError::ProviderTransform {
            provider: ProviderId::from("bedrock"),
            message: format!("unsupported reasoning_effort `{other}` for bedrock provider"),
        }),
    }
}

fn add_response_format_tool(
    provider: &ProviderId,
    value: Value,
    params: &mut ChatParameterMap,
) -> SigmaResult<()> {
    let format = serde_json::from_value::<ResponseFormat>(value).map_err(|err| {
        SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: err.to_string(),
        }
    })?;
    let schema = match format {
        ResponseFormat::Text => return Ok(()),
        ResponseFormat::JsonObject => json!({"type": "object", "properties": {}}),
        ResponseFormat::JsonSchema { json_schema } => json_schema
            .schema
            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    };
    let mut tools = params
        .remove("tools")
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    tools.push(json!({
        "type": "function",
        "function": {
            "name": JSON_TOOL_NAME,
            "description": "Return a JSON object matching the requested schema.",
            "parameters": schema
        }
    }));
    params.insert("tools".to_string(), Value::Array(tools));
    params.insert(
        "tool_choice".to_string(),
        json!({"type": "function", "function": {"name": JSON_TOOL_NAME}}),
    );
    Ok(())
}

fn map_tools(
    provider: &ProviderId,
    tools: Option<Value>,
    tool_choice: Option<Value>,
) -> SigmaResult<(Option<Value>, HashMap<String, String>)> {
    let Some(tools) = tools else {
        return Ok((None, HashMap::new()));
    };
    let tools = tools
        .as_array()
        .cloned()
        .ok_or_else(|| SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: "bedrock tools must be an array".to_string(),
        })?;
    let mut reverse_tool_map = HashMap::new();
    let mut bedrock_tools = Vec::new();

    for tool in tools {
        if is_native_bedrock_tool(&tool) {
            bedrock_tools.push(tool);
            continue;
        }
        let tool = serde_json::from_value::<ToolDefinition>(tool).map_err(|err| {
            SigmaError::ProviderTransform {
                provider: provider.clone(),
                message: err.to_string(),
            }
        })?;
        match tool {
            ToolDefinition::Function(tool) => {
                bedrock_tools.push(tool_spec(tool.function, &mut reverse_tool_map));
            }
            ToolDefinition::Custom(_) => {
                return Err(SigmaError::ProviderTransform {
                    provider: provider.clone(),
                    message: "bedrock converse provider does not support custom tools".to_string(),
                });
            }
        }
    }

    if bedrock_tools.is_empty() {
        return Ok((None, reverse_tool_map));
    }

    let mut tool_config = Map::new();
    tool_config.insert("tools".to_string(), Value::Array(bedrock_tools));
    if let Some(tool_choice) = tool_choice
        && let Some(choice) = bedrock_tool_choice(provider, tool_choice, &mut reverse_tool_map)?
    {
        tool_config.insert("toolChoice".to_string(), choice);
    }

    Ok((Some(Value::Object(tool_config)), reverse_tool_map))
}

fn is_native_bedrock_tool(value: &Value) -> bool {
    value.as_object().is_some_and(|object| {
        object.contains_key("toolSpec")
            || object.contains_key("systemTool")
            || object.contains_key("cachePoint")
    })
}

fn tool_spec(function: FunctionObject, reverse_tool_map: &mut HashMap<String, String>) -> Value {
    let name = bedrock_tool_name(&function.name);
    if name != function.name {
        reverse_tool_map.insert(name.clone(), function.name.clone());
    }
    let schema = sanitize_schema(
        function
            .parameters
            .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
    );
    let mut spec = json!({
        "toolSpec": {
            "name": name,
            "inputSchema": {"json": schema}
        }
    });
    if let Some(description) = function.description
        && let Some(object) = spec.get_mut("toolSpec").and_then(Value::as_object_mut)
    {
        object.insert("description".to_string(), Value::String(description));
    }
    spec
}

fn sanitize_schema(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut sanitized = Map::new();
            for (key, value) in object {
                if matches!(
                    key.as_str(),
                    "$id" | "$schema" | "additionalProperties" | "strict"
                ) {
                    continue;
                }
                sanitized.insert(key, sanitize_schema(value));
            }
            Value::Object(sanitized)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(sanitize_schema).collect()),
        other => other,
    }
}

fn bedrock_tool_choice(
    provider: &ProviderId,
    value: Value,
    reverse_tool_map: &mut HashMap<String, String>,
) -> SigmaResult<Option<Value>> {
    let choice = serde_json::from_value::<ToolChoice>(value).map_err(|err| {
        SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: err.to_string(),
        }
    })?;
    Ok(match choice {
        ToolChoice::Mode(ToolChoiceMode::Auto) => Some(json!({"auto": {}})),
        ToolChoice::Mode(ToolChoiceMode::Required) => Some(json!({"any": {}})),
        ToolChoice::Mode(ToolChoiceMode::None) => None,
        ToolChoice::Function(choice) => {
            let name = bedrock_tool_name(&choice.function.name);
            if name != choice.function.name {
                reverse_tool_map.insert(name.clone(), choice.function.name);
            }
            Some(json!({"tool": {"name": name}}))
        }
        ToolChoice::Allowed(choice) => {
            if matches!(
                choice.mode,
                crate::types::chat::ToolChoiceAllowedMode::Required
            ) {
                Some(json!({"any": {}}))
            } else {
                Some(json!({"auto": {}}))
            }
        }
        ToolChoice::Custom(_) => {
            return Err(SigmaError::ProviderTransform {
                provider: provider.clone(),
                message: "bedrock converse provider does not support custom tool choice"
                    .to_string(),
            });
        }
    })
}

fn bedrock_tool_name(input: &str) -> String {
    let mut output = String::new();
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            output.push(ch);
        } else {
            output.push('_');
        }
    }
    if output
        .chars()
        .next()
        .is_some_and(|ch| !ch.is_ascii_alphabetic())
    {
        output.insert(0, 'a');
    }
    output
}

fn service_tier_block(value: Value) -> Value {
    match value {
        Value::String(value) => json!({"type": value}),
        other => other,
    }
}

fn is_nova_model(model: &str) -> bool {
    model.to_ascii_lowercase().contains("nova")
}

fn is_nova_2_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("amazon.nova-2-") || model.contains("nova-2/")
}

fn encode_model_id(model: &str) -> String {
    let mut encoded = String::new();
    for byte in model.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

pub(super) fn provider_state(reverse_tool_map: HashMap<String, String>, region: &str) -> Value {
    let mut state = Map::new();
    state.insert(
        SIGNING_REGION_STATE_KEY.to_string(),
        Value::String(region.to_string()),
    );
    if !reverse_tool_map.is_empty() {
        state.insert(TOOL_NAME_MAP_STATE_KEY.to_string(), json!(reverse_tool_map));
    }
    Value::Object(state)
}

pub(super) fn reverse_tool_map(context: &ChatAdapterContext<'_>) -> HashMap<String, String> {
    context
        .provider_state
        .as_ref()
        .and_then(|state| state.get(TOOL_NAME_MAP_STATE_KEY))
        .cloned()
        .and_then(|value| serde_json::from_value::<HashMap<String, String>>(value).ok())
        .unwrap_or_default()
}
