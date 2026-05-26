use std::collections::HashMap;

use http::StatusCode;
use serde_json::{Map, Value, json};

use crate::types::chat::{
    Annotation, ChatChoice, ChatResponse, ChatResponseMessage, FinishReason, FunctionToolCall,
    HostedToolUsage, ReasoningBlock, Role, ToolCall, UrlCitation, Usage,
};
use crate::types::shared::{CompletionTokensDetails, FunctionCall, PromptTokensDetails};
use crate::{ChatAdapterContext, SigmaError, SigmaResult};

use super::error::error_from_body;
use super::request::provider_context_block;
use super::state::{current_unix_timestamp, reverse_tool_map};

pub(super) fn anthropic_response_to_chat(
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

pub(super) fn reasoning_response_block(block: &Value) -> ReasoningBlock {
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

pub(super) fn server_tool_use(value: &Value) -> Option<HostedToolUsage> {
    let object = value.as_object()?;
    Some(HostedToolUsage {
        web_search_requests: optional_nonzero(u32_field(object, "web_search_requests")),
        tool_search_requests: optional_nonzero(u32_field(object, "tool_search_requests")),
    })
}

pub(super) fn optional_nonzero(value: u32) -> Option<u32> {
    if value == 0 { None } else { Some(value) }
}

pub(super) fn u32_field(object: &Map<String, Value>, key: &str) -> u32 {
    object
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0)
}

pub(super) fn map_finish_reason(value: &str) -> FinishReason {
    match value {
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "stop_sequence" | "end_turn" => FinishReason::Stop,
        _ => FinishReason::Stop,
    }
}
