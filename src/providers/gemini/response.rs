use http::HeaderMap;
use serde_json::Value;

use crate::types::chat::{
    ChatChoice, ChatResponse, ChatResponseMessage, FinishReason, FunctionToolCall, ReasoningBlock,
    Role, ToolCall,
};
use crate::types::shared::FunctionCall;
use crate::{ChatAdapterContext, SigmaResult};

use super::helpers::{gemini_service_tier_from_headers, map_gemini_finish_reason};
use super::stream::{gemini_usage, grounding_annotations};

pub(super) fn gemini_response_to_chat_response(
    context: &ChatAdapterContext<'_>,
    headers: HeaderMap,
    body: Value,
) -> SigmaResult<ChatResponse> {
    let id = body
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("chatcmpl_gemini")
        .to_string();
    let usage = body.get("usageMetadata").map(gemini_usage);
    let service_tier = gemini_service_tier_from_headers(&headers);
    let choices = gemini_choices(&body, false)?;

    Ok(ChatResponse {
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
            message: ChatResponseMessage {
                content: None,
                reasoning: None,
                refusal: None,
                tool_calls: None,
                annotations: None,
                role: Role::Assistant,
                audio: None,
                provider_context: None,
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
                thought_signatures.push(signature.to_string());
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
                let reasoning = part
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    .map(|signature| vec![ReasoningBlock::signature(signature)]);
                tool_calls.push(ToolCall::Function(FunctionToolCall {
                    id: format!("call_gemini_{idx}_{part_idx}"),
                    function: FunctionCall { name, arguments },
                    reasoning,
                }));
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
    let mut reasoning = Vec::new();
    if !reasoning_content.is_empty() {
        reasoning.push(ReasoningBlock::text(reasoning_content, None::<String>));
    }
    reasoning.extend(
        thought_signatures
            .into_iter()
            .map(ReasoningBlock::signature),
    );

    Ok(ChatChoice {
        index: candidate
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(idx as u64) as u32,
        message: ChatResponseMessage {
            content: (!content.is_empty()).then_some(content),
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            refusal: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            annotations,
            role: Role::Assistant,
            audio: None,
            provider_context: None,
        },
        finish_reason,
        logprobs: None,
    })
}
