use std::pin::Pin;

use bytes::Bytes;
use futures_core::Stream;
use http::{HeaderMap, Method, StatusCode};

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
/// Adapters build this after translating messages, mapping parameters, and
/// selecting an endpoint. The same adapter then signs it into a
/// [`SignedProviderRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRequest {
    /// HTTP method to use for the provider request.
    pub method: Method,
    /// Absolute provider URL.
    pub url: String,
    /// Request headers before provider signing.
    pub headers: HeaderMap,
    /// Serialized provider request body.
    pub body: Bytes,
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
    /// Serialized provider request body.
    pub body: Bytes,
}

impl From<ProviderRequest> for SignedProviderRequest {
    fn from(value: ProviderRequest) -> Self {
        Self {
            method: value.method,
            url: value.url,
            headers: value.headers,
            body: value.body,
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
