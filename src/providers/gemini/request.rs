use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::config::ChatParameterMap;
use crate::providers::common::parse_data_uri;
use crate::types::chat::{
    AssistantContent, ChatMessage, ReasoningBlock, TextContent, ToolCall, ToolContent, UserContent,
    UserContentPart, VideoMetadata,
};
use crate::types::shared::{ImageDetail, ReasoningEffort};
use crate::{ChatAdapterContext, ModelName, ProviderId, SigmaError, SigmaResult};

use super::helpers::{
    add_property_ordering, gemini_service_tier, gemini_video_metadata,
    highest_media_resolution_level, insert_clone, insert_float, is_gemini_3_or_newer,
    is_gemini_file_uri, media_resolution, modalities, parse_function_arguments, remove_schema_key,
    stop_sequences, supports_response_json_schema,
};

pub(super) fn gemini_request_body(
    provider: &ProviderId,
    context: ChatAdapterContext<'_>,
    messages: &[ChatMessage],
    params: &ChatParameterMap,
) -> SigmaResult<Map<String, Value>> {
    let translated = translate_gemini_messages(provider, context.provider_model, messages)?;
    let mut body = Map::new();
    body.insert("contents".to_string(), Value::Array(translated.contents));
    if let Some(system_instruction) = translated.system_instruction {
        body.insert("systemInstruction".to_string(), system_instruction);
    }

    let mapped = map_gemini_params(provider, context.provider_model, params)?;
    let mut generation_config = mapped.generation_config;
    if context.provider_model.as_str().contains("gemini-2")
        && let Some(level) = highest_media_resolution_level(messages)
    {
        let config = generation_config.get_or_insert_with(|| Value::Object(Map::new()));
        if let Value::Object(config) = config {
            config.insert(
                "mediaResolution".to_string(),
                Value::String(level.to_string()),
            );
        }
    }
    if let Some(generation_config) = generation_config {
        body.insert("generationConfig".to_string(), generation_config);
    }
    if let Some(tools) = mapped.tools {
        body.insert("tools".to_string(), tools);
    }
    if let Some(tool_config) = mapped.tool_config {
        body.insert("toolConfig".to_string(), tool_config);
    }
    if let Some(service_tier) = mapped.service_tier {
        body.insert("serviceTier".to_string(), service_tier);
    }

    Ok(body)
}

struct TranslatedGeminiMessages {
    system_instruction: Option<Value>,
    contents: Vec<Value>,
}

fn translate_gemini_messages(
    provider: &ProviderId,
    model: &ModelName,
    messages: &[ChatMessage],
) -> SigmaResult<TranslatedGeminiMessages> {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut last_tool_names = HashMap::<String, String>::new();

    for message in messages {
        match message {
            ChatMessage::Developer(message) => {
                system_parts.extend(developer_content_parts(&message.content));
            }
            ChatMessage::System(message) => {
                system_parts.extend(system_content_parts(&message.content));
            }
            ChatMessage::User(message) => {
                let parts = user_content_parts(provider, model, &message.content)?;
                if !parts.is_empty() {
                    contents.push(json!({
                        "role": "user",
                        "parts": ensure_text_part(parts),
                    }));
                }
            }
            ChatMessage::Assistant(message) => {
                let mut parts = Vec::new();
                if let Some(content) = &message.content {
                    parts.extend(assistant_content_parts(content));
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        if let Some(part) = assistant_tool_call_part(tool_call)? {
                            if let ToolCall::Function(function_call) = tool_call {
                                last_tool_names.insert(
                                    function_call.id.clone(),
                                    function_call.function.name.clone(),
                                );
                            }
                            parts.push(part);
                        }
                    }
                }
                if !parts.is_empty() {
                    contents.push(json!({
                        "role": "model",
                        "parts": parts,
                    }));
                }
            }
            ChatMessage::Tool(message) => {
                let name = last_tool_names.get(&message.tool_call_id).ok_or_else(|| {
                    SigmaError::ProviderTransform {
                        provider: provider.clone(),
                        message: format!(
                            "missing matching assistant tool call for tool response `{}`",
                            message.tool_call_id
                        ),
                    }
                })?;
                contents.push(json!({
                    "role": "user",
                    "parts": [tool_response_part(name, &message.content)],
                }));
            }
        }
    }

    if contents.is_empty() {
        contents.push(json!({
            "role": "user",
            "parts": [{"text": " "}]
        }));
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(json!({ "parts": system_parts }))
    };

    Ok(TranslatedGeminiMessages {
        system_instruction,
        contents,
    })
}

fn developer_content_parts(content: &TextContent) -> Vec<Value> {
    match content {
        TextContent::Text(text) => vec![json!({ "text": text })],
        TextContent::Parts(parts) => parts
            .iter()
            .map(|part| json!({ "text": part.text }))
            .collect(),
    }
}

fn system_content_parts(content: &TextContent) -> Vec<Value> {
    match content {
        TextContent::Text(text) => vec![json!({ "text": text })],
        TextContent::Parts(parts) => parts
            .iter()
            .map(|part| json!({ "text": part.text }))
            .collect(),
    }
}

fn user_content_parts(
    provider: &ProviderId,
    model: &ModelName,
    content: &UserContent,
) -> SigmaResult<Vec<Value>> {
    match content {
        UserContent::Text(text) => Ok(vec![json!({ "text": text })]),
        UserContent::Parts(parts) => parts
            .iter()
            .map(|part| user_content_part(provider, model, part))
            .collect(),
    }
}

fn user_content_part(
    provider: &ProviderId,
    model: &ModelName,
    part: &UserContentPart,
) -> SigmaResult<Value> {
    match part {
        UserContentPart::Text(text) => Ok(json!({ "text": text.text })),
        UserContentPart::Image(image) => gemini_media_part(
            provider,
            model,
            &image.image.url,
            None,
            image.image.detail.as_ref(),
            None,
        ),
        UserContentPart::Audio(audio) => {
            let format = match audio.input_audio.format {
                crate::types::chat::InputAudioFormat::Wav => "audio/wav",
                crate::types::chat::InputAudioFormat::Mp3 => "audio/mp3",
            };
            Ok(json!({
                "inlineData": {
                    "mimeType": format,
                    "data": audio.input_audio.data,
                }
            }))
        }
        UserContentPart::File(file) => {
            let source = file
                .file
                .data
                .as_deref()
                .or(file.file.id.as_deref())
                .ok_or_else(|| SigmaError::ProviderTransform {
                    provider: provider.clone(),
                    message: "gemini file content requires data or id".to_string(),
                })?;
            gemini_media_part(
                provider,
                model,
                source,
                file.file.media_type.as_deref(),
                file.file.detail.as_ref(),
                file.file.video_metadata.as_ref(),
            )
        }
    }
}

fn assistant_content_parts(content: &AssistantContent) -> Vec<Value> {
    match content {
        AssistantContent::Text(text) => vec![json!({ "text": text })],
        AssistantContent::Parts(parts) => parts
            .iter()
            .filter_map(|part| match part {
                crate::types::chat::AssistantContentPart::Text(text) => {
                    Some(json!({ "text": text.text }))
                }
                crate::types::chat::AssistantContentPart::Refusal(_) => None,
            })
            .collect(),
    }
}

fn assistant_tool_call_part(tool_call: &ToolCall) -> SigmaResult<Option<Value>> {
    match tool_call {
        ToolCall::Function(tool_call) => {
            let args = parse_function_arguments(&tool_call.function.arguments);
            let mut part = json!({
                "functionCall": {
                    "name": tool_call.function.name,
                    "args": args,
                }
            });
            if let Some(signature) = tool_call
                .reasoning
                .as_ref()
                .and_then(|blocks| blocks.iter().find_map(ReasoningBlock::signature_value))
                && let Some(object) = part.as_object_mut()
            {
                object.insert(
                    "thoughtSignature".to_string(),
                    Value::String(signature.to_string()),
                );
            }
            Ok(Some(part))
        }
        ToolCall::Custom(_) => Ok(None),
    }
}

fn tool_response_part(name: &str, content: &ToolContent) -> Value {
    let text = match content {
        ToolContent::Text(text) => text.clone(),
        ToolContent::Parts(parts) => parts
            .iter()
            .map(|part| part.text.as_str())
            .collect::<String>(),
    };
    let response = if text.trim_start().starts_with('{') {
        serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({ "content": text }))
    } else {
        json!({ "content": text })
    };

    json!({
        "functionResponse": {
            "name": name,
            "response": response,
        }
    })
}

fn ensure_text_part(mut parts: Vec<Value>) -> Vec<Value> {
    if !parts
        .iter()
        .any(|part| part.get("text").is_some_and(|text| !text.is_null()))
    {
        parts.push(json!({ "text": " " }));
    }
    parts
}

fn gemini_media_part(
    provider: &ProviderId,
    model: &ModelName,
    source: &str,
    format: Option<&str>,
    detail: Option<&ImageDetail>,
    video_metadata: Option<&VideoMetadata>,
) -> SigmaResult<Value> {
    let mut part = if let Some((mime_type, data)) = parse_data_uri(source) {
        json!({
            "inlineData": {
                "mimeType": mime_type,
                "data": data,
            }
        })
    } else if let Some(format) = format
        && !source.starts_with("http://")
        && !source.starts_with("https://")
        && !source.starts_with("gs://")
    {
        json!({
            "inlineData": {
                "mimeType": format,
                "data": source,
            }
        })
    } else if is_gemini_file_uri(source) || source.starts_with("gs://") {
        let mut file_data = Map::new();
        file_data.insert("fileUri".to_string(), Value::String(source.to_string()));
        if let Some(format) = format {
            file_data.insert("mimeType".to_string(), Value::String(format.to_string()));
        }
        json!({ "fileData": file_data })
    } else {
        return Err(SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: "gemini media inputs must be data URIs, raw base64 with an explicit format, Gemini Files API URIs, or gs:// URIs".to_string(),
        });
    };

    if is_gemini_3_or_newer(model.as_str())
        && let Some(media_resolution) = media_resolution(detail)
        && let Some(object) = part.as_object_mut()
    {
        object.insert("mediaResolution".to_string(), media_resolution);
    }
    if let Some(video_metadata) = video_metadata
        && let Some(metadata) = gemini_video_metadata(video_metadata)
        && let Some(object) = part.as_object_mut()
    {
        object.insert("videoMetadata".to_string(), metadata);
    }

    Ok(part)
}

#[derive(Default)]
struct MappedGeminiParams {
    generation_config: Option<Value>,
    tools: Option<Value>,
    tool_config: Option<Value>,
    service_tier: Option<Value>,
}

fn map_gemini_params(
    provider: &ProviderId,
    model: &ModelName,
    params: &ChatParameterMap,
) -> SigmaResult<MappedGeminiParams> {
    let mut generation_config = Map::new();
    let mut function_declarations = Vec::new();
    let mut special_tools = Vec::new();
    let mut tool_config = None;
    let mut service_tier = None;

    for (key, value) in params {
        match key.as_str() {
            "stream" => {}
            "temperature" => insert_float(&mut generation_config, "temperature", value),
            "top_p" => insert_float(&mut generation_config, "topP", value),
            "max_tokens" | "max_completion_tokens" => {
                insert_clone(&mut generation_config, "maxOutputTokens", value);
            }
            "count" => insert_clone(&mut generation_config, "candidateCount", value),
            "stop" => {
                generation_config.insert("stopSequences".to_string(), stop_sequences(value));
            }
            "logprobs" => insert_clone(&mut generation_config, "responseLogprobs", value),
            "top_logprobs" => insert_clone(&mut generation_config, "logprobs", value),
            "frequency_penalty" if !is_gemini_3_or_newer(model.as_str()) => {
                insert_float(&mut generation_config, "frequencyPenalty", value);
            }
            "presence_penalty" if !is_gemini_3_or_newer(model.as_str()) => {
                insert_float(&mut generation_config, "presencePenalty", value);
            }
            "output_modalities" => {
                generation_config.insert("responseModalities".to_string(), modalities(value));
            }
            "audio_output" => {
                generation_config
                    .insert("speechConfig".to_string(), speech_config(provider, value)?);
                generation_config
                    .entry("responseModalities")
                    .or_insert_with(|| json!(["AUDIO"]));
            }
            "response_format" => {
                apply_response_format(model, value, &mut generation_config)?;
            }
            "reasoning_effort" => {
                generation_config.insert(
                    "thinkingConfig".to_string(),
                    reasoning_config(provider, model, value)?,
                );
            }
            "tools" => {
                let mapped = map_tools(value);
                function_declarations.extend(mapped.function_declarations);
                special_tools.extend(mapped.special_tools);
            }
            "tool_choice" => {
                tool_config = map_tool_choice(value);
            }
            "web_search" => special_tools.push(json!({ "googleSearch": {} })),
            "service_tier" => {
                if let Some(value) = value.as_str() {
                    service_tier = Some(Value::String(gemini_service_tier(value).to_string()));
                }
            }
            "parallel_tool_calls" => {}
            _ => {}
        }
    }

    if !function_declarations.is_empty() {
        special_tools.retain(|tool| {
            !tool
                .as_object()
                .is_some_and(|object| object.contains_key("googleSearch"))
        });
    }
    let mut tools = Vec::new();
    if !function_declarations.is_empty() {
        tools.push(json!({ "functionDeclarations": function_declarations }));
    }
    tools.extend(special_tools);

    Ok(MappedGeminiParams {
        generation_config: (!generation_config.is_empty())
            .then_some(Value::Object(generation_config)),
        tools: (!tools.is_empty()).then_some(Value::Array(tools)),
        tool_config,
        service_tier,
    })
}

struct MappedTools {
    function_declarations: Vec<Value>,
    special_tools: Vec<Value>,
}

fn map_tools(value: &Value) -> MappedTools {
    let mut function_declarations = Vec::new();
    let mut special_tools = Vec::new();

    let Some(tools) = value.as_array() else {
        return MappedTools {
            function_declarations,
            special_tools,
        };
    };

    for tool in tools {
        let Some(object) = tool.as_object() else {
            continue;
        };

        if object.get("type").and_then(Value::as_str) == Some("function") {
            if let Some(function) = object.get("function").and_then(Value::as_object) {
                function_declarations.push(function_declaration(function));
            }
        } else if object
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|tool_type| tool_type == "web_search" || tool_type == "web_search_preview")
        {
            special_tools.push(json!({ "googleSearch": {} }));
        } else if object.contains_key("googleSearch") || object.contains_key("google_search") {
            special_tools.push(json!({ "googleSearch": object.get("googleSearch").or_else(|| object.get("google_search")).cloned().unwrap_or_else(|| json!({})) }));
        } else if object.contains_key("urlContext") || object.contains_key("url_context") {
            special_tools.push(json!({ "urlContext": object.get("urlContext").or_else(|| object.get("url_context")).cloned().unwrap_or_else(|| json!({})) }));
        } else if object.contains_key("codeExecution") || object.contains_key("code_execution") {
            special_tools.push(json!({ "codeExecution": object.get("codeExecution").or_else(|| object.get("code_execution")).cloned().unwrap_or_else(|| json!({})) }));
        }
    }

    MappedTools {
        function_declarations,
        special_tools,
    }
}

fn function_declaration(function: &Map<String, Value>) -> Value {
    let mut declaration = Map::new();
    if let Some(name) = function.get("name") {
        declaration.insert("name".to_string(), name.clone());
    }
    if let Some(description) = function.get("description") {
        declaration.insert("description".to_string(), description.clone());
    }
    if let Some(parameters) = function.get("parameters") {
        let mut parameters = parameters.clone();
        remove_schema_key(&mut parameters, "strict");
        remove_schema_key(&mut parameters, "additionalProperties");
        declaration.insert("parameters".to_string(), parameters);
    }
    Value::Object(declaration)
}

fn map_tool_choice(value: &Value) -> Option<Value> {
    match value {
        Value::String(choice) => Some(tool_choice_mode(choice, None)),
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("function") {
                let name = object
                    .get("function")
                    .and_then(Value::as_object)
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str);
                Some(tool_choice_mode("required", name))
            } else {
                object
                    .get("type")
                    .and_then(Value::as_str)
                    .map(|mode| tool_choice_mode(mode, None))
            }
        }
        _ => None,
    }
}

fn tool_choice_mode(mode: &str, allowed_name: Option<&str>) -> Value {
    let gemini_mode = match mode {
        "none" => "NONE",
        "required" => "ANY",
        _ => "AUTO",
    };
    let mut config = Map::new();
    config.insert("mode".to_string(), Value::String(gemini_mode.to_string()));
    if let Some(name) = allowed_name {
        config.insert("allowedFunctionNames".to_string(), json!([name]));
    }
    json!({ "functionCallingConfig": config })
}

fn apply_response_format(
    model: &ModelName,
    value: &Value,
    generation_config: &mut Map<String, Value>,
) -> SigmaResult<()> {
    let Some(object) = value.as_object() else {
        return Ok(());
    };
    match object.get("type").and_then(Value::as_str) {
        Some("json_object") => {
            generation_config.insert(
                "responseMimeType".to_string(),
                Value::String("application/json".to_string()),
            );
        }
        Some("text") => {
            generation_config.insert(
                "responseMimeType".to_string(),
                Value::String("text/plain".to_string()),
            );
        }
        Some("json_schema") => {
            let schema = object
                .get("json_schema")
                .and_then(Value::as_object)
                .and_then(|schema| schema.get("schema"))
                .cloned()
                .unwrap_or(Value::Null);
            if !schema.is_null() {
                generation_config.insert(
                    "responseMimeType".to_string(),
                    Value::String("application/json".to_string()),
                );
                let mut schema = schema;
                remove_schema_key(&mut schema, "strict");
                if supports_response_json_schema(model.as_str()) {
                    generation_config.insert("responseJsonSchema".to_string(), schema);
                } else {
                    remove_schema_key(&mut schema, "additionalProperties");
                    add_property_ordering(&mut schema);
                    generation_config.insert("responseSchema".to_string(), schema);
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn reasoning_config(provider: &ProviderId, model: &ModelName, value: &Value) -> SigmaResult<Value> {
    let effort = serde_json::from_value::<ReasoningEffort>(value.clone()).map_err(|err| {
        SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: err.to_string(),
        }
    })?;
    if is_gemini_3_or_newer(model.as_str()) {
        Ok(reasoning_level_config(model.as_str(), effort))
    } else {
        Ok(reasoning_budget_config(model.as_str(), effort))
    }
}

fn reasoning_budget_config(model: &str, effort: ReasoningEffort) -> Value {
    let (budget, include_thoughts) = match effort {
        ReasoningEffort::None => (0, false),
        ReasoningEffort::Minimal if model.contains("gemini-2.5-flash-lite") => (512, true),
        ReasoningEffort::Minimal if model.contains("gemini-2.5-flash") => (1, true),
        ReasoningEffort::Minimal if model.contains("gemini-2.5-pro") => (128, true),
        ReasoningEffort::Minimal => (128, true),
        ReasoningEffort::Low => (1024, true),
        ReasoningEffort::Medium => (2048, true),
        ReasoningEffort::High => (4096, true),
        ReasoningEffort::Xhigh => (8192, true),
        ReasoningEffort::Max => (16384, true),
    };
    json!({ "thinkingBudget": budget, "includeThoughts": include_thoughts })
}

fn reasoning_level_config(model: &str, effort: ReasoningEffort) -> Value {
    let is_flash = model.contains("gemini-3-flash") || model.contains("gemini-3.1-flash");
    let (level, include_thoughts) = match effort {
        ReasoningEffort::None if is_flash => ("minimal", false),
        ReasoningEffort::None => ("low", false),
        ReasoningEffort::Minimal if is_flash => ("minimal", true),
        ReasoningEffort::Minimal => ("low", true),
        ReasoningEffort::Low => ("low", true),
        ReasoningEffort::Medium if is_flash || model.contains("gemini-3.1-pro-preview") => {
            ("medium", true)
        }
        ReasoningEffort::Medium
        | ReasoningEffort::High
        | ReasoningEffort::Xhigh
        | ReasoningEffort::Max => ("high", true),
    };
    json!({ "thinkingLevel": level, "includeThoughts": include_thoughts })
}

fn speech_config(provider: &ProviderId, value: &Value) -> SigmaResult<Value> {
    let Some(object) = value.as_object() else {
        return Ok(json!({}));
    };
    if let Some(format) = object.get("format").and_then(Value::as_str)
        && format != "pcm16"
    {
        return Err(SigmaError::ProviderTransform {
            provider: provider.clone(),
            message: format!(
                "unsupported audio format for Gemini TTS models: {format}; expected pcm16"
            ),
        });
    }
    if let Some(voice) = object.get("voice").and_then(Value::as_str) {
        Ok(json!({
            "voiceConfig": {
                "prebuiltVoiceConfig": {
                    "voiceName": voice
                }
            }
        }))
    } else {
        Ok(json!({}))
    }
}
