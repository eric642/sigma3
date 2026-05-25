use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use http::{HeaderMap, Method, StatusCode};
use serde_json::Value;

use crate::SigmaResult;

/// Byte stream returned by provider HTTP execution.
///
/// Each item is one raw provider stream frame or byte chunk. The provider's
/// [`crate::ChatCompletionAdapter::transform_stream`] hook converts these bytes
/// into OpenAI-compatible chat stream chunks.
pub type ProviderByteStream = Pin<Box<dyn Stream<Item = SigmaResult<Bytes>> + Send + 'static>>;

/// Provider endpoint selected by a chat adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderEndpoint {
    /// HTTP method to use for the provider request.
    pub method: Method,
    /// Absolute provider URL.
    pub url: String,
}

/// Unsigned provider request produced by an adapter.
///
/// Adapters build this after mapping parameters and selecting an endpoint.
/// The same adapter then signs it into a
/// [`SignedProviderRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRequest {
    /// HTTP method to use for the provider request.
    pub method: Method,
    /// Absolute provider URL.
    pub url: String,
    /// Request headers before provider signing.
    pub headers: HeaderMap,
    /// Structured JSON provider request body.
    ///
    /// Adapters construct JSON here so tests and signing hooks can inspect the
    /// final provider-native body before sigma serializes it for HTTP.
    pub body: Value,
    /// Provider-local state for response or stream transformation.
    ///
    /// This is never sent over HTTP. Adapters use it for request-scoped data
    /// such as Anthropic tool-name reverse maps.
    pub provider_state: Option<Value>,
}

/// Provider request after authentication or provider-specific signing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedProviderRequest {
    /// HTTP method to use for the provider request.
    pub method: Method,
    /// Absolute provider URL.
    pub url: String,
    /// Request headers after provider signing.
    pub headers: HeaderMap,
    /// Structured JSON provider request body.
    ///
    /// sigma serializes this value when sending the HTTP request.
    pub body: Value,
    /// Provider-local state for response or stream transformation.
    ///
    /// This is never sent over HTTP.
    pub provider_state: Option<Value>,
}

impl From<ProviderRequest> for SignedProviderRequest {
    fn from(value: ProviderRequest) -> Self {
        Self {
            method: value.method,
            url: value.url,
            headers: value.headers,
            body: value.body,
            provider_state: value.provider_state,
        }
    }
}

/// Raw provider response returned by sigma's HTTP execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderResponse {
    /// Provider HTTP status.
    pub status: StatusCode,
    /// Provider response headers.
    pub headers: HeaderMap,
    /// Raw provider response body.
    pub body: Bytes,
}
