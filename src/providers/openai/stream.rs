use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;
use serde_json::Value;

use crate::provider_http::ProviderByteStream;
use crate::providers::common::event_data;
use crate::types::chat::ChatStreamChunk;
use crate::{ProviderId, SigmaError, SigmaResult};

use super::config::OpenAiFlavor;
use super::response::{map_stream_reasoning_content, sanitize_null_usage_tokens};

pub(super) struct OpenAiSseStream {
    provider: ProviderId,
    stream: ProviderByteStream,
    buffer: String,
    pending: VecDeque<SigmaResult<ChatStreamChunk>>,
    done: bool,
    flavor: OpenAiFlavor,
}

impl OpenAiSseStream {
    pub(super) fn new(
        provider: ProviderId,
        stream: ProviderByteStream,
        flavor: OpenAiFlavor,
    ) -> Self {
        Self {
            provider,
            stream,
            buffer: String::new(),
            pending: VecDeque::new(),
            done: false,
            flavor,
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

        self.drain_raw_json_lines();

        if flush {
            let event = self.buffer.trim().to_string();
            self.buffer.clear();
            if !event.is_empty() {
                self.push_event(&event);
            }
        }
    }

    fn drain_raw_json_lines(&mut self) {
        loop {
            let Some(index) = self.buffer.find('\n') else {
                return;
            };
            let line = self.buffer[..index].trim();
            if !line.starts_with('{') && line != "[DONE]" {
                return;
            }

            let event = line.to_string();
            self.buffer.drain(..index + 1);
            self.push_event(&event);
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

        if self.flavor.sanitizes_usage() {
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
