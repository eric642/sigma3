use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use http::StatusCode;
use serde_json::{Map, Value};

use crate::provider_http::ProviderByteStream;
use crate::providers::common::current_unix_timestamp;
use crate::types::chat::{
    AssistantDelta, ChatStreamChoice, ChatStreamChunk, FinishReason, FunctionCallDelta,
    ReasoningBlock, Role, ToolCallDelta, ToolCallKind, Usage,
};
use crate::{ProviderId, SigmaError, SigmaResult};

use super::response::{bedrock_usage, business_stream_error, map_finish_reason, stream_error};

pub(super) struct BedrockConverseStream {
    provider: ProviderId,
    stream: ProviderByteStream,
    pending: VecDeque<SigmaResult<ChatStreamChunk>>,
    done: bool,
    model: String,
    buffer: Vec<u8>,
    reverse_tool_map: HashMap<String, String>,
    id: String,
    created: u32,
}

impl BedrockConverseStream {
    pub(super) fn new(
        provider: ProviderId,
        model: String,
        stream: ProviderByteStream,
        reverse_tool_map: HashMap<String, String>,
    ) -> Self {
        Self {
            provider,
            stream,
            pending: VecDeque::new(),
            done: false,
            model,
            buffer: Vec::new(),
            reverse_tool_map,
            id: "chatcmpl_bedrock_stream".to_string(),
            created: current_unix_timestamp(),
        }
    }

    fn push_chunk(&mut self, chunk: Bytes) {
        self.buffer.extend_from_slice(&chunk);
        self.drain_buffer();
    }

    fn drain_buffer(&mut self) {
        loop {
            match next_event_stream_message(&mut self.buffer) {
                Ok(Some(message)) => self.handle_event_message(message),
                Ok(None) => return,
                Err(err) => {
                    self.done = true;
                    self.pending
                        .push_back(Err(stream_error(&self.provider, err)));
                    return;
                }
            }
            if self.done {
                return;
            }
        }
    }

    fn handle_event_message(&mut self, message: EventStreamMessage) {
        let message_type = message.headers.get(":message-type").map(String::as_str);
        if matches!(message_type, Some("exception" | "error")) {
            self.done = true;
            self.pending
                .push_back(Err(self.exception_error(&message.headers, &message.payload)));
            return;
        }

        let value = match serde_json::from_slice::<Value>(&message.payload) {
            Ok(value) => value,
            Err(err) => {
                self.done = true;
                self.pending
                    .push_back(Err(stream_error(&self.provider, err.to_string())));
                return;
            }
        };
        self.handle_stream_value(&value);
    }

    fn exception_error(&self, headers: &HashMap<String, String>, payload: &[u8]) -> SigmaError {
        let details = serde_json::from_slice::<Value>(payload).ok();
        let code = headers
            .get(":exception-type")
            .cloned()
            .or_else(|| headers.get(":error-code").cloned());
        let message = details
            .as_ref()
            .and_then(|value| value.get("message").or_else(|| value.get("Message")))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| String::from_utf8_lossy(payload).into_owned());

        business_stream_error(
            &self.provider,
            StatusCode::INTERNAL_SERVER_ERROR,
            code,
            message,
            details,
        )
    }

    fn handle_stream_value(&mut self, value: &Value) {
        if let Some(message_start) = value.get("messageStart").and_then(Value::as_object) {
            if let Some(conversation_id) =
                message_start.get("conversationId").and_then(Value::as_str)
            {
                self.id = format!("chatcmpl-{conversation_id}");
            }
            return;
        }

        if let Some(start) = value.get("contentBlockStart").and_then(Value::as_object) {
            self.handle_content_block_start(start);
            return;
        }

        if let Some(delta) = value.get("contentBlockDelta").and_then(Value::as_object) {
            self.handle_content_block_delta(delta);
            return;
        }

        if value.get("contentBlockStop").is_some() {
            return;
        }

        if let Some(message_stop) = value.get("messageStop").and_then(Value::as_object) {
            let finish_reason = message_stop
                .get("stopReason")
                .and_then(Value::as_str)
                .map(map_finish_reason);
            self.pending
                .push_back(Ok(self.chunk(None, None, finish_reason, None)));
            return;
        }

        if let Some(metadata) = value.get("metadata").and_then(Value::as_object) {
            let usage = metadata
                .get("usage")
                .and_then(Value::as_object)
                .map(bedrock_usage);
            if usage.is_some() {
                self.pending
                    .push_back(Ok(self.chunk(None, None, None, usage)));
            }
        }
    }

    fn handle_content_block_start(&mut self, value: &Map<String, Value>) {
        let index = content_block_index(value);
        let Some(start) = value.get("start").and_then(Value::as_object) else {
            return;
        };
        if let Some(tool_use) = start.get("toolUse").and_then(Value::as_object) {
            let id = tool_use
                .get("toolUseId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let name = tool_use
                .get("name")
                .and_then(Value::as_str)
                .map(|name| {
                    self.reverse_tool_map
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| name.to_string())
                })
                .unwrap_or_default();
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
            )));
        }
        if let Some(reasoning) = start.get("reasoningContent").and_then(Value::as_object) {
            self.pending.push_back(Ok(self
                .chunk(None, None, None, None)
                .with_reasoning(reasoning_delta(reasoning))));
        }
    }

    fn handle_content_block_delta(&mut self, value: &Map<String, Value>) {
        let index = content_block_index(value);
        let Some(delta) = value.get("delta").and_then(Value::as_object) else {
            return;
        };
        if let Some(text) = delta.get("text").and_then(Value::as_str) {
            self.pending
                .push_back(Ok(self.chunk(Some(text.to_string()), None, None, None)));
            return;
        }
        if let Some(tool_use) = delta.get("toolUse").and_then(Value::as_object) {
            let arguments = tool_use
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            self.pending.push_back(Ok(self.chunk(
                None,
                Some(vec![ToolCallDelta {
                    index,
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
            )));
            return;
        }
        if let Some(reasoning) = delta.get("reasoningContent").and_then(Value::as_object) {
            self.pending.push_back(Ok(self
                .chunk(None, None, None, None)
                .with_reasoning(reasoning_delta(reasoning))));
        }
    }

    fn chunk(
        &self,
        content: Option<String>,
        tool_calls: Option<Vec<ToolCallDelta>>,
        finish_reason: Option<FinishReason>,
        usage: Option<Usage>,
    ) -> ChatStreamChunk {
        let choices = if content.is_none() && tool_calls.is_none() && finish_reason.is_none() {
            Vec::new()
        } else {
            vec![ChatStreamChoice {
                index: 0,
                delta: AssistantDelta {
                    content,
                    reasoning: None,
                    tool_calls,
                    role: Some(Role::Assistant),
                    refusal: None,
                },
                finish_reason,
                logprobs: None,
            }]
        };

        ChatStreamChunk {
            id: self.id.clone(),
            choices,
            created: self.created,
            model: self.model.clone(),
            service_tier: None,
            object: "chat.completion.chunk".to_string(),
            usage,
        }
    }
}

trait WithReasoning {
    fn with_reasoning(self, reasoning: Option<Vec<ReasoningBlock>>) -> Self;
}

impl WithReasoning for ChatStreamChunk {
    fn with_reasoning(mut self, reasoning: Option<Vec<ReasoningBlock>>) -> Self {
        if let Some(choice) = self.choices.first_mut() {
            choice.delta.reasoning = reasoning;
        } else if reasoning.is_some() {
            self.choices.push(ChatStreamChoice {
                index: 0,
                delta: AssistantDelta {
                    content: None,
                    reasoning,
                    tool_calls: None,
                    role: Some(Role::Assistant),
                    refusal: None,
                },
                finish_reason: None,
                logprobs: None,
            });
        }
        self
    }
}

impl Stream for BedrockConverseStream {
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
                Poll::Ready(Some(Err(err))) => {
                    self.done = true;
                    return Poll::Ready(Some(Err(err)));
                }
                Poll::Ready(None) => {
                    self.done = true;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

struct EventStreamMessage {
    headers: HashMap<String, String>,
    payload: Vec<u8>,
}

fn next_event_stream_message(buffer: &mut Vec<u8>) -> Result<Option<EventStreamMessage>, String> {
    if buffer.len() < 12 {
        return Ok(None);
    }

    let total_len = read_u32(&buffer[0..4]) as usize;
    let headers_len = read_u32(&buffer[4..8]) as usize;
    if total_len < 16 || headers_len > total_len.saturating_sub(16) {
        return Err("invalid bedrock event-stream frame length".to_string());
    }
    if buffer.len() < total_len {
        return Ok(None);
    }

    let expected_prelude_crc = read_u32(&buffer[8..12]);
    let actual_prelude_crc = crc32fast::hash(&buffer[0..8]);
    if expected_prelude_crc != actual_prelude_crc {
        return Err("invalid bedrock event-stream prelude crc".to_string());
    }
    let expected_message_crc = read_u32(&buffer[total_len - 4..total_len]);
    let actual_message_crc = crc32fast::hash(&buffer[0..total_len - 4]);
    if expected_message_crc != actual_message_crc {
        return Err("invalid bedrock event-stream message crc".to_string());
    }

    let headers_start = 12;
    let payload_start = headers_start + headers_len;
    let payload_end = total_len - 4;
    let headers = parse_headers(&buffer[headers_start..payload_start])?;
    let payload = buffer[payload_start..payload_end].to_vec();
    buffer.drain(..total_len);

    Ok(Some(EventStreamMessage { headers, payload }))
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn parse_headers(bytes: &[u8]) -> Result<HashMap<String, String>, String> {
    let mut headers = HashMap::new();
    let mut index = 0;

    while index < bytes.len() {
        let name_len = bytes[index] as usize;
        index += 1;
        if index + name_len + 1 > bytes.len() {
            return Err("invalid bedrock event-stream header".to_string());
        }
        let name = std::str::from_utf8(&bytes[index..index + name_len])
            .map_err(|err| err.to_string())?
            .to_string();
        index += name_len;
        let value_type = bytes[index];
        index += 1;
        if value_type != 7 {
            return Err(format!(
                "unsupported bedrock event-stream header type `{value_type}`"
            ));
        }
        if index + 2 > bytes.len() {
            return Err("invalid bedrock event-stream string header".to_string());
        }
        let value_len = u16::from_be_bytes([bytes[index], bytes[index + 1]]) as usize;
        index += 2;
        if index + value_len > bytes.len() {
            return Err("invalid bedrock event-stream string header length".to_string());
        }
        let value = std::str::from_utf8(&bytes[index..index + value_len])
            .map_err(|err| err.to_string())?
            .to_string();
        index += value_len;
        headers.insert(name, value);
    }

    Ok(headers)
}

fn content_block_index(value: &Map<String, Value>) -> u32 {
    value
        .get("contentBlockIndex")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0)
}

fn reasoning_delta(reasoning: &Map<String, Value>) -> Option<Vec<ReasoningBlock>> {
    if let Some(text) = reasoning.get("text").and_then(Value::as_str) {
        return Some(vec![ReasoningBlock::text(
            text.to_string(),
            reasoning
                .get("signature")
                .and_then(Value::as_str)
                .map(str::to_string),
        )]);
    }
    if let Some(reasoning_text) = reasoning.get("reasoningText").and_then(Value::as_object) {
        return Some(vec![ReasoningBlock::text(
            reasoning_text
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            reasoning_text
                .get("signature")
                .and_then(Value::as_str)
                .map(str::to_string),
        )]);
    }
    reasoning
        .get("redactedContent")
        .and_then(Value::as_str)
        .map(|value| vec![ReasoningBlock::redacted(value.to_string(), None::<String>)])
}
