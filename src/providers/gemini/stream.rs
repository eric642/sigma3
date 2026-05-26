use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use serde_json::Value;

use crate::provider_http::ProviderByteStream;
use crate::providers::common::event_data;
use crate::types::chat::{
    Annotation, AssistantDelta, ChatStreamChoice, ChatStreamChunk, FinishReason, ReasoningBlock,
    Role, ToolCallDelta, ToolCallKind, UrlCitation, Usage,
};
use crate::types::shared::{CompletionTokensDetails, PromptTokensDetails};
use crate::{ModelName, ProviderId, SigmaError, SigmaResult};

use super::helpers::{map_gemini_finish_reason, u32_field, u32_value};

pub(super) struct GeminiSseStream {
    provider: ProviderId,
    model: ModelName,
    stream: ProviderByteStream,
    buffer: String,
    pending: VecDeque<SigmaResult<ChatStreamChunk>>,
    done: bool,
    seen_tool_calls: bool,
}

impl GeminiSseStream {
    pub(super) fn new(provider: ProviderId, model: ModelName, stream: ProviderByteStream) -> Self {
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
        let Some(data) = event_data(event, true, false) else {
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

    fn stream_response_from_value(&mut self, value: Value) -> SigmaResult<ChatStreamChunk> {
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

        Ok(ChatStreamChunk {
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

fn stream_choice_from_candidate(candidate: &Value, idx: usize) -> SigmaResult<ChatStreamChoice> {
    let parts = candidate
        .get("content")
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array);
    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut tool_calls = Vec::new();
    let mut reasoning = Vec::new();

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
                reasoning.push(ReasoningBlock::signature(signature));
            }
            if let Some(function_call) = part.get("functionCall").and_then(Value::as_object) {
                let function = crate::types::chat::FunctionCallDelta {
                    name: function_call
                        .get("name")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    arguments: function_call.get("args").map(|args| {
                        serde_json::to_string(args).unwrap_or_else(|_| "null".to_string())
                    }),
                };
                tool_calls.push(ToolCallDelta {
                    index: part_idx as u32,
                    id: Some(format!("call_gemini_{idx}_{part_idx}")),
                    r#type: Some(ToolCallKind::Function),
                    function: Some(function),
                    reasoning: (!reasoning.is_empty()).then_some(reasoning.clone()),
                });
            }
        }
    }
    if !reasoning_content.is_empty() {
        reasoning.insert(0, ReasoningBlock::text(reasoning_content, None::<String>));
    }

    Ok(ChatStreamChoice {
        index: candidate
            .get("index")
            .and_then(Value::as_u64)
            .unwrap_or(idx as u64) as u32,
        delta: AssistantDelta {
            content: (!content.is_empty()).then_some(content),
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            role: parts.map(|_| Role::Assistant),
            refusal: None,
        },
        finish_reason: candidate
            .get("finishReason")
            .and_then(Value::as_str)
            .map(map_gemini_finish_reason),
        logprobs: None,
    })
}

pub(super) fn gemini_usage(value: &Value) -> Usage {
    let prompt_tokens = u32_field(value, "promptTokenCount");
    let candidates_tokens = u32_field(value, "candidatesTokenCount");
    let reasoning_tokens = value.get("thoughtsTokenCount").and_then(u32_value);
    let completion_tokens = candidates_tokens + reasoning_tokens.unwrap_or(0);
    let total_tokens = u32_field(value, "totalTokenCount");
    let cached_tokens = value.get("cachedContentTokenCount").and_then(u32_value);
    let prompt_details = modality_details(value.get("promptTokensDetails"));
    let completion_details = modality_details(value.get("candidatesTokensDetails"));

    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: cached_tokens,
        hosted_tool_use: None,
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

pub(super) fn grounding_annotations(metadata: &Value, _content: &str) -> Option<Vec<Annotation>> {
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
            annotations.push(Annotation::UrlCitation {
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
