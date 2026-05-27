use std::collections::HashMap;

use http::StatusCode;
use serde_json::{Map, Value};

use crate::providers::common::current_unix_timestamp;
use crate::types::chat::{
    ChatChoice, ChatResponse, ChatResponseMessage, FinishReason, FunctionToolCall, ReasoningBlock,
    Role, ServiceTier, ToolCall, Usage,
};
use crate::types::shared::{CompletionTokensDetails, FunctionCall, PromptTokensDetails};
use crate::{ChatAdapterContext, ProviderId, SigmaError, SigmaResult};

use super::request::reverse_tool_map;

pub(super) fn bedrock_response_to_chat(
    context: &ChatAdapterContext<'_>,
    body: Value,
) -> SigmaResult<ChatResponse> {
    let message = body
        .get("output")
        .and_then(|output| output.get("message"))
        .and_then(Value::as_object)
        .ok_or_else(|| SigmaError::ProviderResponse {
            provider: context.provider.to_owned(),
            message: "bedrock response missing output.message".to_string(),
        })?;
    let reverse_map = reverse_tool_map(context);
    let content = message
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let translated = translate_content(content, &reverse_map);
    let usage = body
        .get("usage")
        .and_then(Value::as_object)
        .map(bedrock_usage);

    Ok(ChatResponse {
        id: body
            .get("responseId")
            .and_then(Value::as_str)
            .unwrap_or("chatcmpl_bedrock")
            .to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatResponseMessage {
                content: (!translated.text.is_empty()).then_some(translated.text),
                reasoning: (!translated.reasoning.is_empty()).then_some(translated.reasoning),
                refusal: None,
                tool_calls: (!translated.tool_calls.is_empty()).then_some(translated.tool_calls),
                annotations: None,
                role: Role::Assistant,
                audio: None,
                provider_context: None,
            },
            finish_reason: body
                .get("stopReason")
                .and_then(Value::as_str)
                .map(map_finish_reason),
            logprobs: None,
        }],
        created: current_unix_timestamp(),
        model: context.provider_model.to_string(),
        service_tier: body.get("serviceTier").and_then(service_tier),
        object: "chat.completion".to_string(),
        usage,
    })
}

struct TranslatedContent {
    text: String,
    reasoning: Vec<ReasoningBlock>,
    tool_calls: Vec<ToolCall>,
}

fn translate_content(
    content: &[Value],
    reverse_map: &HashMap<String, String>,
) -> TranslatedContent {
    let mut text = String::new();
    let mut reasoning = Vec::new();
    let mut tool_calls = Vec::new();

    for block in content {
        if let Some(value) = block.get("text").and_then(Value::as_str) {
            text.push_str(value);
        }
        if let Some(tool_use) = block.get("toolUse").and_then(Value::as_object) {
            tool_calls.push(tool_use_to_tool_call(tool_use, reverse_map));
        }
        if let Some(reasoning_content) = block.get("reasoningContent").and_then(Value::as_object) {
            collect_reasoning(reasoning_content, &mut reasoning);
        }
    }

    TranslatedContent {
        text,
        reasoning,
        tool_calls,
    }
}

fn tool_use_to_tool_call(
    tool_use: &Map<String, Value>,
    reverse_map: &HashMap<String, String>,
) -> ToolCall {
    let name = tool_use
        .get("name")
        .and_then(Value::as_str)
        .map(|name| {
            reverse_map
                .get(name)
                .cloned()
                .unwrap_or_else(|| name.to_string())
        })
        .unwrap_or_default();
    let arguments = tool_use
        .get("input")
        .map(|value| serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()))
        .unwrap_or_else(|| "{}".to_string());

    ToolCall::Function(FunctionToolCall {
        id: tool_use
            .get("toolUseId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        function: FunctionCall { name, arguments },
        reasoning: None,
    })
}

fn collect_reasoning(reasoning_content: &Map<String, Value>, reasoning: &mut Vec<ReasoningBlock>) {
    if let Some(reasoning_text) = reasoning_content.get("reasoningText") {
        let text = reasoning_text
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let signature = reasoning_text.get("signature").and_then(Value::as_str);
        reasoning.push(ReasoningBlock::text(
            text.to_string(),
            signature.map(str::to_string),
        ));
    }
    if let Some(redacted) = reasoning_content
        .get("redactedContent")
        .and_then(Value::as_str)
    {
        reasoning.push(ReasoningBlock::redacted(
            redacted.to_string(),
            None::<String>,
        ));
    }
}

pub(super) fn bedrock_usage(usage: &Map<String, Value>) -> Usage {
    let raw_input = u32_field(usage, "inputTokens");
    let cache_read = u32_field(usage, "cacheReadInputTokens");
    let cache_creation = u32_field(usage, "cacheWriteInputTokens");
    let prompt_tokens = raw_input + cache_read + cache_creation;
    let completion_tokens = u32_field(usage, "outputTokens");

    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        cache_creation_input_tokens: optional_nonzero(cache_creation),
        cache_read_input_tokens: optional_nonzero(cache_read),
        hosted_tool_use: None,
        inference_geo: None,
        speed: None,
        prompt_tokens_details: Some(PromptTokensDetails {
            audio_tokens: None,
            cached_tokens: optional_nonzero(cache_read),
            text_tokens: optional_nonzero(raw_input),
            image_tokens: None,
            video_tokens: None,
        }),
        completion_tokens_details: Some(CompletionTokensDetails {
            accepted_prediction_tokens: None,
            audio_tokens: None,
            text_tokens: optional_nonzero(completion_tokens),
            image_tokens: None,
            video_tokens: None,
            reasoning_tokens: None,
            rejected_prediction_tokens: None,
        }),
    }
}

pub(super) fn u32_field(object: &Map<String, Value>, name: &str) -> u32 {
    object
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0)
}

pub(super) fn optional_nonzero(value: u32) -> Option<u32> {
    (value != 0).then_some(value)
}

pub(super) fn map_finish_reason(value: &str) -> FinishReason {
    match value {
        "max_tokens" | "length" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "content_filtered" | "guardrail_intervened" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

fn service_tier(value: &Value) -> Option<ServiceTier> {
    let value = value
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| value.as_str())?;
    match value {
        "auto" => Some(ServiceTier::Auto),
        "default" => Some(ServiceTier::Default),
        "flex" => Some(ServiceTier::Flex),
        "scale" => Some(ServiceTier::Scale),
        "priority" => Some(ServiceTier::Priority),
        _ => None,
    }
}

pub(super) fn bedrock_error_response(
    context: &ChatAdapterContext<'_>,
    response: crate::ProviderResponse,
) -> SigmaError {
    let details = serde_json::from_slice::<Value>(&response.body).ok();
    let code = details
        .as_ref()
        .and_then(|body| {
            body.get("__type")
                .or_else(|| body.get("code"))
                .or_else(|| body.get("Code"))
        })
        .and_then(Value::as_str)
        .map(str::to_string);
    let message = details
        .as_ref()
        .and_then(|body| {
            body.get("message")
                .or_else(|| body.get("Message"))
                .or_else(|| body.get("error"))
        })
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| fallback_error_message(response.status, &response.body));
    let retry_after = crate::providers::common::parse_retry_after(&response.headers);

    if let Some(err) = crate::providers::common::classify_provider_error(
        context.provider,
        response.status,
        code.as_deref(),
        &message,
        retry_after,
        details.clone(),
    ) {
        return err;
    }

    SigmaError::ProviderBusiness {
        provider: context.provider.to_owned(),
        status: response.status,
        code,
        message,
        details,
    }
}

pub(super) fn stream_error(provider: &ProviderId, message: impl Into<String>) -> SigmaError {
    SigmaError::ProviderResponse {
        provider: provider.clone(),
        message: message.into(),
    }
}

pub(super) fn business_stream_error(
    provider: &ProviderId,
    status: StatusCode,
    code: Option<String>,
    message: String,
    details: Option<Value>,
) -> SigmaError {
    SigmaError::ProviderBusiness {
        provider: provider.clone(),
        status,
        code,
        message,
        details,
    }
}

fn fallback_error_message(status: StatusCode, body: &[u8]) -> String {
    if body.is_empty() {
        status
            .canonical_reason()
            .unwrap_or("bedrock returned unsuccessful HTTP status")
            .to_string()
    } else {
        String::from_utf8_lossy(body).into_owned()
    }
}
