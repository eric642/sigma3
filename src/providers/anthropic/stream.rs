use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use serde_json::Value;

use crate::provider_http::ProviderByteStream;
use crate::providers::common::{SseLineBuffer, event_data};
use crate::types::chat::{
    AssistantDelta, ChatStreamChoice, ChatStreamChunk, FinishReason, FunctionCallDelta,
    ReasoningBlock, ToolCallDelta, ToolCallKind, Usage,
};
use crate::types::shared::{CompletionTokensDetails, PromptTokensDetails};
use crate::{ProviderId, SigmaError, SigmaResult};

use crate::providers::common::current_unix_timestamp;

use super::RESPONSE_FORMAT_TOOL_NAME;
use super::error::stream_error_from_event;
use super::response::{
    map_finish_reason, optional_nonzero, reasoning_response_block, server_tool_use, u32_field,
};

pub(super) struct AnthropicSseStream {
    provider: ProviderId,
    stream: ProviderByteStream,
    buffer: SseLineBuffer,
    pending: std::collections::VecDeque<SigmaResult<ChatStreamChunk>>,
    done: bool,
    id: String,
    model: String,
    created: u32,
    prompt_tokens: u32,
    current_tool_index: Option<u32>,
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
    /// Set while the active tool block is sigma's `json_tool_call` fallback,
    /// so `input_json_delta` events emit `delta.content` instead of tool-call
    /// deltas.
    current_tool_is_response_format_fallback: bool,
    /// Set if the stream observed at least one fallback tool block, so the
    /// terminal `message_delta` rewrites `tool_use` finish reason to `Stop`.
    response_format_fallback_hit: bool,
    response_format_fallback_active: bool,
    reverse_tool_map: HashMap<String, String>,
}

impl AnthropicSseStream {
    pub(super) fn new(
        provider: ProviderId,
        stream: ProviderByteStream,
        reverse_tool_map: HashMap<String, String>,
        response_format_fallback_active: bool,
    ) -> Self {
        Self {
            provider,
            stream,
            buffer: SseLineBuffer::new(),
            pending: std::collections::VecDeque::new(),
            done: false,
            id: "msg_anthropic_stream".to_string(),
            model: String::new(),
            created: current_unix_timestamp(),
            prompt_tokens: 0,
            current_tool_index: None,
            current_tool_id: None,
            current_tool_name: None,
            current_tool_is_response_format_fallback: false,
            response_format_fallback_hit: false,
            response_format_fallback_active,
            reverse_tool_map,
        }
    }

    fn push_chunk(&mut self, chunk: Bytes) {
        self.buffer.extend(&chunk);
        self.drain_buffer(false);
    }

    fn drain_buffer(&mut self, flush: bool) {
        while let Some(event) = self.buffer.next_event() {
            self.push_event(&event);
            if self.done {
                return;
            }
        }
        if flush && !self.buffer.is_empty() {
            let event = self.buffer.drain_remaining();
            let event = event.trim();
            if !event.is_empty() {
                let event = event.to_string();
                self.push_event(&event);
            }
        }
    }

    fn push_event(&mut self, event: &str) {
        let Some(data) = event_data(event, false, false) else {
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
                self.current_tool_is_response_format_fallback = false;
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
                let error_type = error
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                self.done = true;
                self.pending.push_back(Err(stream_error_from_event(
                    &self.provider,
                    error_type.as_deref(),
                    &message,
                    Some(error),
                )));
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
                let raw_name = block.get("name").and_then(Value::as_str);
                let is_response_format_fallback = self.response_format_fallback_active
                    && raw_name == Some(RESPONSE_FORMAT_TOOL_NAME);
                let name = raw_name
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
                self.current_tool_is_response_format_fallback = is_response_format_fallback;

                if is_response_format_fallback {
                    self.response_format_fallback_hit = true;
                    // Skip emitting a tool-call delta; later input_json_delta
                    // events arrive as content fragments instead.
                    return;
                }
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
                if self.current_tool_is_response_format_fallback {
                    // Surface the partial JSON as content so the caller's
                    // streaming consumer can assemble the JSON object the same
                    // way it would for a non-fallback model.
                    self.pending
                        .push_back(Ok(self.chunk(Some(arguments), None, None, None, None)));
                    return;
                }
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
        let raw_finish_reason = value
            .get("delta")
            .and_then(|delta| delta.get("stop_reason"))
            .and_then(Value::as_str)
            .map(map_finish_reason);
        let finish_reason = if self.response_format_fallback_hit {
            // Anthropic stops with `tool_use` when our injected
            // `json_tool_call` finishes streaming. Rewriting it to `Stop`
            // matches the semantics the caller asked for via response_format.
            Some(FinishReason::Stop)
        } else {
            raw_finish_reason
        };
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
