use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use serde_json::Value;

use crate::provider_http::ProviderByteStream;
use crate::providers::common::{SseLineBuffer, event_data};
use crate::types::chat::ChatStreamChunk;
use crate::{ProviderId, SigmaError, SigmaResult};

use super::response::{map_stream_reasoning_content, sanitize_null_usage_tokens};

pub(super) struct OpenAiSseStream {
    provider: ProviderId,
    stream: ProviderByteStream,
    buffer: SseLineBuffer,
    pending: VecDeque<SigmaResult<ChatStreamChunk>>,
    done: bool,
    sanitize_null_usage_tokens: bool,
}

impl OpenAiSseStream {
    pub(super) fn new(
        provider: ProviderId,
        stream: ProviderByteStream,
        sanitize_null_usage_tokens: bool,
    ) -> Self {
        Self {
            provider,
            stream,
            buffer: SseLineBuffer::new(),
            pending: VecDeque::new(),
            done: false,
            sanitize_null_usage_tokens,
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

        self.drain_raw_json_lines();

        if flush {
            let event = self.buffer.drain_remaining();
            let event = event.trim();
            if !event.is_empty() {
                let event = event.to_string();
                self.push_event(&event);
            }
        }
    }

    fn drain_raw_json_lines(&mut self) {
        // OpenAI-compatible servers sometimes emit JSON-lines streams that lack
        // the `event:`/`data:` framing. Peek the first line and only consume it
        // when it looks like a full JSON object or `[DONE]`; everything else
        // must keep waiting for the `\n\n` event terminator above.
        loop {
            let line = {
                let head = self.buffer.peek();
                let Some(idx) = head.find('\n') else {
                    return;
                };
                head[..idx].trim().to_string()
            };
            if !line.starts_with('{') && line != "[DONE]" {
                return;
            }
            self.buffer.next_line();
            self.push_event(&line);
            if self.done {
                return;
            }
        }
    }

    fn push_event(&mut self, event: &str) {
        let Some(data) = event_data(event, true, true) else {
            return;
        };

        if data == "[DONE]" {
            self.done = true;
            return;
        }

        let mut value = match serde_json::from_str::<Value>(&data) {
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

        if self.sanitize_null_usage_tokens {
            sanitize_null_usage_tokens(&mut value);
        }
        map_stream_reasoning_content(&mut value);

        let chunk = serde_json::from_value(value).map_err(|err| SigmaError::ProviderResponse {
            provider: self.provider.clone(),
            message: err.to_string(),
        });
        self.pending.push_back(chunk);
    }
}

impl Stream for OpenAiSseStream {
    type Item = SigmaResult<ChatStreamChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(item) = self.pending.pop_front() {
            return Poll::Ready(Some(item));
        }

        if self.done {
            return Poll::Ready(None);
        }

        loop {
            let poll = self.stream.as_mut().poll_next(cx);
            match poll {
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
