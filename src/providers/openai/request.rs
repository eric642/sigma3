use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::config::ChatParameterMap;
use crate::types::chat::{
    AssistantContent, AssistantContentPart, ChatMessage, DeveloperMessage, FileInput,
    SystemMessage, TextContent, ToolCall, ToolContent, UserContent, UserContentPart,
};
use crate::{ModelName, ProviderId, SigmaError, SigmaResult};

pub(super) fn chat_completions_url(api_base: &str) -> String {
    let api_base = api_base.trim_end_matches('/');

    if api_base.ends_with("/chat/completions") {
        api_base.to_string()
    } else {
        format!("{api_base}/chat/completions")
    }
}

fn is_generated_body_key(key: &str) -> bool {
    key == "model" || key == "messages"
}

fn contains_provider_option(provider_options: Option<&ChatParameterMap>, key: &str) -> bool {
    provider_options.is_some_and(|provider_options| provider_options.contains_key(key))
}

pub(super) fn rename_param(params: &mut ChatParameterMap, from: &str, to: &str) {
    if let Some(value) = params.remove(from) {
        params.insert(to.to_string(), value);
    }
}

pub(super) fn openai_chat_body(
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
