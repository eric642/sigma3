use std::collections::{HashMap, VecDeque};
use std::env;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use http::header::CONTENT_TYPE;
use http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Number, Value, json};

use crate::config::{ChatParameterMap, SecretString};
use crate::provider_http::{
    ProviderByteStream, ProviderEndpoint, ProviderRequest, ProviderResponse, SignedProviderRequest,
};
use crate::types::chat::{
    ChatChoice, ChatChoiceStream, ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallChunk, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionResponseMessage,
    ChatCompletionResponseMessageAnnotation, ChatCompletionStreamResponseDelta, CompletionUsage,
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse, FinishReason, FunctionType,
    Role, StopConfiguration, UrlCitation,
};
use crate::types::shared::{
    CompletionTokensDetails, FunctionCall, ImageDetail, PromptTokensDetails, ReasoningEffort,
};
use crate::{
    ChatAdapterContext, ChatAdapterRequest, ChatCompletionAdapter, ChatStream, ModelName,
    ProviderDriver, ProviderId, ProviderInit, ProviderKind, ProviderKindStatic, SigmaError,
    SigmaResult, StreamBehavior, submit_provider,
};

const GEMINI_KIND: ProviderKindStatic = ProviderKindStatic::new("gemini");
const GEMINI_DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const GEMINI_API_KEY_HEADER: &str = "x-goog-api-key";

const SUPPORTED_GEMINI_CHAT_PARAMS: &[&str] = &[
    "audio",
    "frequency_penalty",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "modalities",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "service_tier",
    "stop",
    "stream",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "web_search_options",
];

struct GeminiProvider {
    id: ProviderId,
    kind: ProviderKind,
    chat: GeminiChatAdapter,
}

impl GeminiProvider {
    fn from_config(init: ProviderInit<GeminiConfig>) -> SigmaResult<Arc<dyn ProviderDriver>> {
        let api_base = init
            .common
            .api_base
            .clone()
            .or_else(|| non_empty_env("GEMINI_API_BASE"))
            .unwrap_or_else(|| GEMINI_DEFAULT_BASE_URL.to_string());
        let api_key = init
            .common
            .api_key
            .clone()
            .or_else(|| non_empty_env("GOOGLE_API_KEY").map(SecretString::from))
            .or_else(|| non_empty_env("GEMINI_API_KEY").map(SecretString::from));
        let headers = header_map_from_config(&init.id, init.common.headers)?;

        if api_key.is_none() && !headers.contains_key(GEMINI_API_KEY_HEADER) {
            return Err(SigmaError::ProviderConfig {
                provider: Some(init.id.clone()),
                message:
                    "gemini provider requires api_key, GOOGLE_API_KEY, GEMINI_API_KEY, or an x-goog-api-key header"
                        .to_string(),
            });
        }

        Ok(Arc::new(Self {
            id: init.id.clone(),
            kind: init.kind,
            chat: GeminiChatAdapter {
                provider: init.id,
                api_base,
                api_key,
                headers,
                api_version: init.config.api_version,
            },
        }))
    }
}

impl ProviderDriver for GeminiProvider {
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
struct GeminiConfig {
    /// Gemini REST API version selection.
    api_version: GeminiApiVersion,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum GeminiApiVersion {
    /// Use v1alpha for Gemini 3 models and v1beta for other Gemini chat models.
    #[default]
    Auto,
    /// Always use the v1 API path.
    V1,
    /// Always use the v1beta API path.
    V1Beta,
    /// Always use the v1alpha API path.
    V1Alpha,
}

impl GeminiApiVersion {
    fn segment(self, model: &ModelName) -> &'static str {
        match self {
            Self::Auto if is_gemini_3_or_newer(model.as_str()) => "v1alpha",
            Self::Auto => "v1beta",
            Self::V1 => "v1",
            Self::V1Beta => "v1beta",
            Self::V1Alpha => "v1alpha",
        }
    }
}

struct GeminiChatAdapter {
    provider: ProviderId,
    api_base: String,
    api_key: Option<SecretString>,
    headers: HeaderMap,
    api_version: GeminiApiVersion,
}

impl ChatCompletionAdapter for GeminiChatAdapter {
    fn supported_openai_params(&self) -> Vec<&'static str> {
        SUPPORTED_GEMINI_CHAT_PARAMS.to_vec()
    }

    fn map_openai_params(&self, params: ChatParameterMap) -> SigmaResult<ChatParameterMap> {
        Ok(params)
    }

    fn validate_environment(&self) -> SigmaResult<()> {
        Ok(())
    }

    fn endpoint(&self, request: &ChatAdapterRequest<'_>) -> SigmaResult<ProviderEndpoint> {
        let stream = request
            .params
            .get("stream")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let endpoint = if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        let mut url = gemini_model_url(
            &self.api_base,
            self.api_version.segment(request.context.provider_model),
            request.context.provider_model,
            endpoint,
        );
        if stream {
            url.push_str("?alt=sse");
        }

        Ok(ProviderEndpoint {
            method: Method::POST,
            url,
        })
    }

    fn transform_request(
        &self,
        request: ChatAdapterRequest<'_>,
        endpoint: ProviderEndpoint,
    ) -> SigmaResult<ProviderRequest> {
        let mut body = gemini_request_body(
            &self.provider,
            request.context,
            request.messages,
            &request.params,
        )?;
        if let Some(provider_options) = request.provider_options {
            for (key, value) in provider_options {
                body.insert(key.clone(), value.clone());
            }
        }

        let mut headers = self.headers.clone();
        if !headers.contains_key(CONTENT_TYPE) {
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        }

        Ok(ProviderRequest {
            method: endpoint.method,
            url: endpoint.url,
            headers,
            body: Value::Object(body),
            provider_state: None,
        })
    }

    fn sign_request(&self, mut request: ProviderRequest) -> SigmaResult<SignedProviderRequest> {
        if !request.headers.contains_key(GEMINI_API_KEY_HEADER)
            && let Some(api_key) = &self.api_key
        {
            let value = HeaderValue::from_str(api_key.expose_secret()).map_err(|err| {
                SigmaError::ProviderSigning {
                    provider: self.provider.clone(),
                    message: err.to_string(),
                }
            })?;
            request
                .headers
                .insert(HeaderName::from_static(GEMINI_API_KEY_HEADER), value);
        }

        Ok(request.into())
    }

    fn transform_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaResult<CreateChatCompletionResponse> {
        let body = parse_response_json(&self.provider, response.body.as_ref())?;
        gemini_response_to_chat_response(context, response.headers, body)
    }

    fn transform_error_response(
        &self,
        context: &ChatAdapterContext<'_>,
        response: ProviderResponse,
    ) -> SigmaError {
        gemini_error_response(context, response)
    }

    fn transform_stream(
        &self,
        context: &ChatAdapterContext<'_>,
        stream: ProviderByteStream,
    ) -> SigmaResult<ChatStream> {
        Ok(Box::pin(GeminiSseStream::new(
            self.provider.clone(),
            context.provider_model.clone(),
            stream,
        )))
    }

    fn stream_behavior(&self) -> StreamBehavior {
        StreamBehavior::native(true)
    }
}

fn gemini_request_body(
    provider: &ProviderId,
    context: ChatAdapterContext<'_>,
    messages: &[ChatCompletionRequestMessage],
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
    messages: &[ChatCompletionRequestMessage],
) -> SigmaResult<TranslatedGeminiMessages> {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();
    let mut last_tool_names = HashMap::<String, String>::new();

    for message in messages {
        match message {
            ChatCompletionRequestMessage::Developer(message) => {
                system_parts.extend(developer_content_parts(&message.content));
            }
            ChatCompletionRequestMessage::System(message) => {
                system_parts.extend(system_content_parts(&message.content));
            }
            ChatCompletionRequestMessage::User(message) => {
                let parts = user_content_parts(provider, model, &message.content)?;
                if !parts.is_empty() {
                    contents.push(json!({
                        "role": "user",
                        "parts": ensure_text_part(parts),
                    }));
                }
            }
            ChatCompletionRequestMessage::Assistant(message) => {
                let mut parts = Vec::new();
                if let Some(content) = &message.content {
                    parts.extend(assistant_content_parts(content));
                }
                if let Some(tool_calls) = &message.tool_calls {
                    for tool_call in tool_calls {
                        if let Some(part) = assistant_tool_call_part(tool_call)? {
                            if let ChatCompletionMessageToolCalls::Function(function_call) =
                                tool_call
                            {
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
            ChatCompletionRequestMessage::Tool(message) => {
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

fn developer_content_parts(content: &ChatCompletionRequestDeveloperMessageContent) -> Vec<Value> {
    match content {
        ChatCompletionRequestDeveloperMessageContent::Text(text) => vec![json!({ "text": text })],
        ChatCompletionRequestDeveloperMessageContent::Array(parts) => parts
            .iter()
            .map(|part| match part {
                crate::types::chat::ChatCompletionRequestDeveloperMessageContentPart::Text(
                    text,
                ) => json!({ "text": text.text }),
            })
            .collect(),
    }
}

fn system_content_parts(content: &ChatCompletionRequestSystemMessageContent) -> Vec<Value> {
    match content {
        ChatCompletionRequestSystemMessageContent::Text(text) => vec![json!({ "text": text })],
        ChatCompletionRequestSystemMessageContent::Array(parts) => parts
            .iter()
            .map(|part| match part {
                crate::types::chat::ChatCompletionRequestSystemMessageContentPart::Text(text) => {
                    json!({ "text": text.text })
                }
            })
            .collect(),
    }
}

fn user_content_parts(
    provider: &ProviderId,
    model: &ModelName,
    content: &ChatCompletionRequestUserMessageContent,
) -> SigmaResult<Vec<Value>> {
    match content {
        ChatCompletionRequestUserMessageContent::Text(text) => Ok(vec![json!({ "text": text })]),
        ChatCompletionRequestUserMessageContent::Array(parts) => parts
            .iter()
            .map(|part| user_content_part(provider, model, part))
            .collect(),
    }
}

fn user_content_part(
    provider: &ProviderId,
    model: &ModelName,
    part: &ChatCompletionRequestUserMessageContentPart,
) -> SigmaResult<Value> {
    match part {
        ChatCompletionRequestUserMessageContentPart::Text(text) => Ok(json!({ "text": text.text })),
        ChatCompletionRequestUserMessageContentPart::ImageUrl(image) => gemini_media_part(
            provider,
            model,
            &image.image_url.url,
            None,
            image.image_url.detail.as_ref(),
            None,
        ),
        ChatCompletionRequestUserMessageContentPart::InputAudio(audio) => {
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
        ChatCompletionRequestUserMessageContentPart::File(file) => {
            let source = file
                .file
                .file_data
                .as_deref()
                .or(file.file.file_id.as_deref())
                .ok_or_else(|| SigmaError::ProviderTransform {
                    provider: provider.clone(),
                    message: "gemini file content requires file_data or file_id".to_string(),
                })?;
            gemini_media_part(
                provider,
                model,
                source,
                file.file.format.as_deref(),
                file.file.detail.as_ref(),
                file.file.video_metadata.as_ref(),
            )
        }
    }
}

fn assistant_content_parts(content: &ChatCompletionRequestAssistantMessageContent) -> Vec<Value> {
    match content {
        ChatCompletionRequestAssistantMessageContent::Text(text) => vec![json!({ "text": text })],
        ChatCompletionRequestAssistantMessageContent::Array(parts) => parts
            .iter()
            .filter_map(|part| match part {
                crate::types::chat::ChatCompletionRequestAssistantMessageContentPart::Text(
                    text,
                ) => Some(json!({ "text": text.text })),
                crate::types::chat::ChatCompletionRequestAssistantMessageContentPart::Refusal(
                    _,
                ) => None,
            })
            .collect(),
    }
}

fn assistant_tool_call_part(
    tool_call: &ChatCompletionMessageToolCalls,
) -> SigmaResult<Option<Value>> {
    match tool_call {
        ChatCompletionMessageToolCalls::Function(tool_call) => {
            let args = parse_function_arguments(&tool_call.function.arguments);
            let mut part = json!({
                "functionCall": {
                    "name": tool_call.function.name,
                    "args": args,
                }
            });
            if let Some(signature) = tool_call
                .provider_specific_fields
                .as_ref()
                .and_then(|fields| fields.get("thought_signature"))
                .and_then(Value::as_str)
                && let Some(object) = part.as_object_mut()
            {
                object.insert(
                    "thoughtSignature".to_string(),
                    Value::String(signature.to_string()),
                );
            }
            Ok(Some(part))
        }
        ChatCompletionMessageToolCalls::Custom(_) => Ok(None),
    }
}

fn tool_response_part(name: &str, content: &ChatCompletionRequestToolMessageContent) -> Value {
    let text = match content {
        ChatCompletionRequestToolMessageContent::Text(text) => text.clone(),
        ChatCompletionRequestToolMessageContent::Array(parts) => parts
            .iter()
            .map(|part| match part {
                crate::types::chat::ChatCompletionRequestToolMessageContentPart::Text(text) => {
                    text.text.as_str()
                }
            })
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
    video_metadata: Option<&Value>,
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
            "n" => insert_clone(&mut generation_config, "candidateCount", value),
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
            "modalities" => {
                generation_config.insert("responseModalities".to_string(), modalities(value));
            }
            "audio" => {
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
            "web_search_options" => special_tools.push(json!({ "googleSearch": {} })),
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

fn gemini_response_to_chat_response(
    context: &ChatAdapterContext<'_>,
    headers: HeaderMap,
    body: Value,
) -> SigmaResult<CreateChatCompletionResponse> {
    let id = body
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("chatcmpl_gemini")
        .to_string();
    let usage = body.get("usageMetadata").map(gemini_usage);
    let service_tier = gemini_service_tier_from_headers(&headers);
    let choices = gemini_choices(&body, false)?;

    Ok(CreateChatCompletionResponse {
        id,
        choices,
        created: 0,
        model: context.provider_model.to_string(),
        service_tier,
        object: "chat.completion".to_string(),
        usage,
    })
}

fn gemini_choices(body: &Value, stream_delta: bool) -> SigmaResult<Vec<ChatChoice>> {
    if let Some(prompt_feedback) = body.get("promptFeedback").and_then(Value::as_object)
        && prompt_feedback.contains_key("blockReason")
    {
        return Ok(vec![ChatChoice {
            index: 0,
            message: ChatCompletionResponseMessage {
                content: None,
                reasoning_content: None,
                refusal: None,
                tool_calls: None,
                annotations: None,
                role: Role::Assistant,
                audio: None,
                thinking_blocks: None,
                provider_specific_fields: None,
            },
            finish_reason: Some(FinishReason::ContentFilter),
            logprobs: None,
        }]);
    }

    let Some(candidates) = body.get("candidates").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    candidates
        .iter()
        .enumerate()
        .map(|(idx, candidate)| gemini_candidate_choice(candidate, idx, stream_delta))
        .collect()
}

fn gemini_candidate_choice(
    candidate: &Value,
    idx: usize,
    _stream_delta: bool,
) -> SigmaResult<ChatChoice> {
    let parts = candidate
        .get("content")
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array);
    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut tool_calls = Vec::new();
    let mut thought_signatures = Vec::new();

    if let Some(parts) = parts {
        for (part_idx, part) in parts.iter().enumerate() {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    reasoning_content.push_str(text);
                } else {
                    content.push_str(text);
                }
            }
            if let Some(signature) = part.get("thoughtSignature").and_then(Value::as_str) {
                thought_signatures.push(Value::String(signature.to_string()));
            }
            if let Some(function_call) = part.get("functionCall").and_then(Value::as_object) {
                let name = function_call
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let arguments = function_call
                    .get("args")
                    .map(|args| serde_json::to_string(args).unwrap_or_else(|_| "null".to_string()))
                    .unwrap_or_else(|| "null".to_string());
                let provider_specific_fields = part
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    .map(|signature| json!({ "thought_signature": signature }));
                tool_calls.push(ChatCompletionMessageToolCalls::Function(
                    ChatCompletionMessageToolCall {
                        id: format!("call_gemini_{idx}_{part_idx}"),
                        function: FunctionCall { name, arguments },
                        provider_specific_fields,
                    },
                ));
            }
        }
    }

    let finish_reason = if !tool_calls.is_empty() {
        Some(FinishReason::ToolCalls)
    } else {
        candidate
            .get("finishReason")
            .and_then(Value::as_str)
            .map(map_gemini_finish_reason)
    };
    let annotations = candidate
        .get("groundingMetadata")
        .and_then(|metadata| grounding_annotations(metadata, content.as_str()));
    let provider_specific_fields = (!thought_signatures.is_empty()).then(|| {
        json!({
            "thought_signatures": thought_signatures
        })
    });

    Ok(ChatChoice {
        index: candidate
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(idx as u64) as u32,
        message: ChatCompletionResponseMessage {
            content: (!content.is_empty()).then_some(content),
            reasoning_content: (!reasoning_content.is_empty()).then_some(reasoning_content),
            refusal: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            annotations,
            role: Role::Assistant,
            audio: None,
            thinking_blocks: None,
            provider_specific_fields,
        },
        finish_reason,
        logprobs: None,
    })
}

struct GeminiSseStream {
    provider: ProviderId,
    model: ModelName,
    stream: ProviderByteStream,
    buffer: String,
    pending: VecDeque<SigmaResult<CreateChatCompletionStreamResponse>>,
    done: bool,
    seen_tool_calls: bool,
}

impl GeminiSseStream {
    fn new(provider: ProviderId, model: ModelName, stream: ProviderByteStream) -> Self {
        Self {
            provider,
            model,
            stream,
            buffer: String::new(),
            pending: VecDeque::new(),
            done: false,
            seen_tool_calls: false,
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
        let chunk = self.stream_response_from_value(value);
        self.pending.push_back(chunk);
    }

    fn stream_response_from_value(
        &mut self,
        value: Value,
    ) -> SigmaResult<CreateChatCompletionStreamResponse> {
        let id = value
            .get("responseId")
            .and_then(Value::as_str)
            .unwrap_or("chatcmpl_gemini")
            .to_string();
        let usage = value.get("usageMetadata").map(gemini_usage);
        let mut choices = Vec::new();

        if let Some(candidates) = value.get("candidates").and_then(Value::as_array) {
            for (idx, candidate) in candidates.iter().enumerate() {
                let mut choice = stream_choice_from_candidate(candidate, idx)?;
                if choice.delta.tool_calls.is_some() {
                    self.seen_tool_calls = true;
                    choice.finish_reason = Some(FinishReason::ToolCalls);
                } else if self.seen_tool_calls && choice.finish_reason == Some(FinishReason::Stop) {
                    choice.finish_reason = Some(FinishReason::ToolCalls);
                }
                choices.push(choice);
            }
        }

        Ok(CreateChatCompletionStreamResponse {
            id,
            choices,
            created: 0,
            model: self.model.to_string(),
            service_tier: None,
            object: "chat.completion.chunk".to_string(),
            usage,
        })
    }
}

impl Stream for GeminiSseStream {
    type Item = SigmaResult<CreateChatCompletionStreamResponse>;

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

fn stream_choice_from_candidate(candidate: &Value, idx: usize) -> SigmaResult<ChatChoiceStream> {
    let parts = candidate
        .get("content")
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array);
    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut tool_calls = Vec::new();
    let mut provider_specific_fields = None;

    if let Some(parts) = parts {
        for (part_idx, part) in parts.iter().enumerate() {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    reasoning_content.push_str(text);
                } else {
                    content.push_str(text);
                }
            }
            if let Some(signature) = part.get("thoughtSignature").and_then(Value::as_str) {
                provider_specific_fields = Some(json!({ "thought_signature": signature }));
            }
            if let Some(function_call) = part.get("functionCall").and_then(Value::as_object) {
                let function = crate::types::chat::FunctionCallStream {
                    name: function_call
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    arguments: function_call.get("args").map(|args| {
                        serde_json::to_string(args).unwrap_or_else(|_| "null".to_string())
                    }),
                };
                tool_calls.push(ChatCompletionMessageToolCallChunk {
                    index: part_idx as u32,
                    id: Some(format!("call_gemini_{idx}_{part_idx}")),
                    r#type: Some(FunctionType::Function),
                    function: Some(function),
                    provider_specific_fields: provider_specific_fields.clone(),
                });
            }
        }
    }

    Ok(ChatChoiceStream {
        index: candidate
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(idx as u64) as u32,
        delta: ChatCompletionStreamResponseDelta {
            content: (!content.is_empty()).then_some(content),
            reasoning_content: (!reasoning_content.is_empty()).then_some(reasoning_content),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            role: parts.map(|_| Role::Assistant),
            refusal: None,
            thinking_blocks: None,
            provider_specific_fields,
        },
        finish_reason: candidate
            .get("finishReason")
            .and_then(Value::as_str)
            .map(map_gemini_finish_reason),
        logprobs: None,
    })
}

fn gemini_usage(value: &Value) -> CompletionUsage {
    let prompt_tokens = u32_field(value, "promptTokenCount");
    let candidates_tokens = u32_field(value, "candidatesTokenCount");
    let reasoning_tokens = value.get("thoughtsTokenCount").and_then(u32_value);
    let completion_tokens = candidates_tokens + reasoning_tokens.unwrap_or(0);
    let total_tokens = u32_field(value, "totalTokenCount");
    let cached_tokens = value.get("cachedContentTokenCount").and_then(u32_value);
    let prompt_details = modality_details(value.get("promptTokensDetails"));
    let completion_details = modality_details(value.get("candidatesTokensDetails"));

    CompletionUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: cached_tokens,
        server_tool_use: None,
        inference_geo: None,
        speed: None,
        prompt_tokens_details: Some(PromptTokensDetails {
            audio_tokens: prompt_details.audio,
            cached_tokens,
            text_tokens: prompt_details.text,
            image_tokens: prompt_details.image,
            video_tokens: prompt_details.video,
        }),
        completion_tokens_details: Some(CompletionTokensDetails {
            accepted_prediction_tokens: None,
            audio_tokens: completion_details.audio,
            text_tokens: completion_details.text,
            image_tokens: completion_details.image,
            video_tokens: completion_details.video,
            reasoning_tokens,
            rejected_prediction_tokens: None,
        }),
    }
}

#[derive(Default)]
struct ModalityCounts {
    text: Option<u32>,
    audio: Option<u32>,
    image: Option<u32>,
    video: Option<u32>,
}

fn modality_details(value: Option<&Value>) -> ModalityCounts {
    let mut counts = ModalityCounts::default();
    let Some(details) = value.and_then(Value::as_array) else {
        return counts;
    };
    for detail in details {
        let Some(modality) = detail.get("modality").and_then(Value::as_str) else {
            continue;
        };
        let Some(count) = detail
            .get("tokenCount")
            .or_else(|| detail.get("token_count"))
            .and_then(u32_value)
        else {
            continue;
        };
        match modality.to_ascii_uppercase().as_str() {
            "TEXT" | "DOCUMENT" => add_count(&mut counts.text, count),
            "AUDIO" => add_count(&mut counts.audio, count),
            "IMAGE" => add_count(&mut counts.image, count),
            "VIDEO" => add_count(&mut counts.video, count),
            _ => {}
        }
    }
    counts
}

fn add_count(target: &mut Option<u32>, count: u32) {
    *target = Some(target.unwrap_or(0) + count);
}

fn grounding_annotations(
    metadata: &Value,
    _content: &str,
) -> Option<Vec<ChatCompletionResponseMessageAnnotation>> {
    let supports = metadata.get("groundingSupports")?.as_array()?;
    let chunks = metadata.get("groundingChunks")?.as_array()?;
    let mut annotations = Vec::new();

    for support in supports {
        let segment = support.get("segment").and_then(Value::as_object);
        let start_index = segment
            .and_then(|segment| segment.get("startIndex"))
            .and_then(u32_value);
        let end_index = segment
            .and_then(|segment| segment.get("endIndex"))
            .and_then(u32_value);
        let chunk_index = support
            .get("groundingChunkIndices")
            .and_then(Value::as_array)
            .and_then(|indices| indices.first())
            .and_then(Value::as_u64)
            .map(|index| index as usize);

        if let (Some(start_index), Some(end_index), Some(chunk_index)) =
            (start_index, end_index, chunk_index)
            && let Some(web) = chunks
                .get(chunk_index)
                .and_then(|chunk| chunk.get("web"))
                .and_then(Value::as_object)
        {
            annotations.push(ChatCompletionResponseMessageAnnotation::UrlCitation {
                url_citation: UrlCitation {
                    start_index,
                    end_index,
                    title: web
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    url: web
                        .get("uri")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                },
            });
        }
    }

    (!annotations.is_empty()).then_some(annotations)
}

fn gemini_error_response(
    context: &ChatAdapterContext<'_>,
    response: ProviderResponse,
) -> SigmaError {
    let body = serde_json::from_slice::<Value>(&response.body).ok();
    let error = body
        .as_ref()
        .and_then(|body| body.get("error"))
        .and_then(Value::as_object);
    let code = error
        .and_then(|error| error.get("status"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            error
                .and_then(|error| error.get("code"))
                .map(|code| code.to_string())
        });
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| fallback_error_message(response.status, &response.body));

    SigmaError::ProviderBusiness {
        provider: context.provider.to_owned(),
        status: response.status,
        code,
        message,
        details: body,
    }
}

fn parse_response_json(provider: &ProviderId, body: &[u8]) -> SigmaResult<Value> {
    serde_json::from_slice(body).map_err(|err| SigmaError::ProviderResponse {
        provider: provider.clone(),
        message: err.to_string(),
    })
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

fn gemini_model_url(api_base: &str, version: &str, model: &ModelName, endpoint: &str) -> String {
    let base = api_base.trim_end_matches('/');
    let versioned_base =
        if base.ends_with("/v1") || base.ends_with("/v1beta") || base.ends_with("/v1alpha") {
            base.to_string()
        } else {
            format!("{base}/{version}")
        };
    let model = if model.as_str().starts_with("models/") {
        model.to_string()
    } else {
        format!("models/{model}")
    };
    format!("{versioned_base}/{model}:{endpoint}")
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

fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

fn insert_clone(map: &mut Map<String, Value>, key: &str, value: &Value) {
    map.insert(key.to_string(), value.clone());
}

fn insert_float(map: &mut Map<String, Value>, key: &str, value: &Value) {
    let Some(number) = value.as_f64() else {
        insert_clone(map, key, value);
        return;
    };
    let rounded = (number * 1_000_000.0).round() / 1_000_000.0;
    let Some(number) = Number::from_f64(rounded) else {
        insert_clone(map, key, value);
        return;
    };
    map.insert(key.to_string(), Value::Number(number));
}

fn stop_sequences(value: &Value) -> Value {
    match serde_json::from_value::<StopConfiguration>(value.clone()) {
        Ok(StopConfiguration::String(value)) => json!([value]),
        Ok(StopConfiguration::StringArray(values)) => json!(values),
        Err(_) => Value::Null,
    }
}

fn modalities(value: &Value) -> Value {
    let Some(values) = value.as_array() else {
        return Value::Null;
    };
    Value::Array(
        values
            .iter()
            .filter_map(Value::as_str)
            .map(|value| match value {
                "text" => "TEXT",
                "audio" => "AUDIO",
                "image" => "IMAGE",
                _ => "MODALITY_UNSPECIFIED",
            })
            .map(|value| Value::String(value.to_string()))
            .collect(),
    )
}

fn gemini_service_tier(value: &str) -> &str {
    match value {
        "auto" => "priority",
        "default" => "standard",
        other => other,
    }
}

fn gemini_service_tier_from_headers(
    headers: &HeaderMap,
) -> Option<crate::types::chat::ServiceTier> {
    let value = headers
        .get("x-gemini-service-tier")
        .and_then(|value| value.to_str().ok())?;
    let value = if value.eq_ignore_ascii_case("standard") {
        "default"
    } else {
        value
    };
    serde_json::from_value(Value::String(value.to_ascii_lowercase())).ok()
}

fn map_gemini_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "MAX_TOKENS" => FinishReason::Length,
        "SAFETY"
        | "RECITATION"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

fn parse_function_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.to_string()))
}

fn parse_data_uri(value: &str) -> Option<(&str, &str)> {
    let value = value.strip_prefix("data:")?;
    let (mime_type, data) = value.split_once(";base64,")?;
    Some((mime_type, data))
}

fn is_gemini_file_uri(value: &str) -> bool {
    value.starts_with("https://generativelanguage.googleapis.com/")
        || value.starts_with("https://www.googleapis.com/")
}

fn media_resolution(detail: Option<&ImageDetail>) -> Option<Value> {
    let level = media_resolution_level(detail?)?;
    Some(json!({ "level": level }))
}

fn highest_media_resolution_level(
    messages: &[ChatCompletionRequestMessage],
) -> Option<&'static str> {
    let mut best = None;

    for message in messages {
        let ChatCompletionRequestMessage::User(message) = message else {
            continue;
        };
        let ChatCompletionRequestUserMessageContent::Array(parts) = &message.content else {
            continue;
        };
        for part in parts {
            let detail = match part {
                ChatCompletionRequestUserMessageContentPart::ImageUrl(image) => {
                    image.image_url.detail.as_ref()
                }
                ChatCompletionRequestUserMessageContentPart::File(file) => {
                    file.file.detail.as_ref()
                }
                ChatCompletionRequestUserMessageContentPart::Text(_)
                | ChatCompletionRequestUserMessageContentPart::InputAudio(_) => None,
            };
            if detail.is_some_and(|detail| {
                media_resolution_level(detail).is_some()
                    && image_detail_priority(detail) > best.map_or(0, image_detail_priority)
            }) {
                best = detail;
            }
        }
    }

    best.and_then(media_resolution_level)
}

fn media_resolution_level(detail: &ImageDetail) -> Option<&'static str> {
    let level = match detail {
        ImageDetail::Low => "MEDIA_RESOLUTION_LOW",
        ImageDetail::Medium => "MEDIA_RESOLUTION_MEDIUM",
        ImageDetail::High => "MEDIA_RESOLUTION_HIGH",
        ImageDetail::UltraHigh | ImageDetail::Original => "MEDIA_RESOLUTION_ULTRA_HIGH",
        ImageDetail::Auto => return None,
    };
    Some(level)
}

fn image_detail_priority(detail: &ImageDetail) -> u8 {
    match detail {
        ImageDetail::Auto => 0,
        ImageDetail::Low => 1,
        ImageDetail::Medium => 2,
        ImageDetail::High => 3,
        ImageDetail::UltraHigh | ImageDetail::Original => 4,
    }
}

fn gemini_video_metadata(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    let mut metadata = Map::new();
    for (key, value) in object {
        match key.as_str() {
            "fps" => {
                metadata.insert("fps".to_string(), value.clone());
            }
            "start_offset" | "startOffset" => {
                metadata.insert("startOffset".to_string(), value.clone());
            }
            "end_offset" | "endOffset" => {
                metadata.insert("endOffset".to_string(), value.clone());
            }
            _ => {}
        }
    }
    (!metadata.is_empty()).then_some(Value::Object(metadata))
}

fn remove_schema_key(value: &mut Value, key: &str) {
    match value {
        Value::Object(object) => {
            object.remove(key);
            for value in object.values_mut() {
                remove_schema_key(value, key);
            }
        }
        Value::Array(values) => {
            for value in values {
                remove_schema_key(value, key);
            }
        }
        _ => {}
    }
}

fn add_property_ordering(value: &mut Value) {
    let Value::Object(object) = value else {
        return;
    };
    if let Some(properties) = object.get_mut("properties").and_then(Value::as_object_mut) {
        let ordering = properties
            .keys()
            .cloned()
            .map(Value::String)
            .collect::<Vec<_>>();
        for value in properties.values_mut() {
            add_property_ordering(value);
        }
        object.insert("propertyOrdering".to_string(), Value::Array(ordering));
    }
    if let Some(items) = object.get_mut("items") {
        add_property_ordering(items);
    }
}

fn supports_response_json_schema(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("gemini-2.") || is_gemini_3_or_newer(&model)
}

fn is_gemini_3_or_newer(model: &str) -> bool {
    model.to_ascii_lowercase().contains("gemini-3")
}

fn u32_field(value: &Value, key: &str) -> u32 {
    value.get(key).and_then(u32_value).unwrap_or_default()
}

fn u32_value(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|value| u32::try_from(value).ok())
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
        } else if line.starts_with('{') {
            data_lines.push(line.to_string());
        }
    }

    (!data_lines.is_empty()).then(|| data_lines.join("\n"))
}

submit_provider! {
    kind: GEMINI_KIND,
    constructor: GeminiProvider::from_config,
    config: GeminiConfig,
}
