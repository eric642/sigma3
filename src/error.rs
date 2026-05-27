//! Errors raised by sigma.

use crate::{DeploymentId, ModelRef, ProviderId, ProviderKind};
use http::StatusCode;
use serde_json::Value;

/// Result type used by sigma public APIs.
pub type SigmaResult<T> = Result<T, SigmaError>;

/// Error type returned by sigma configuration, routing, provider, and HTTP APIs.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SigmaError {
    /// Returned when a builder fails to produce a valid value.
    #[error("invalid args: {0}")]
    InvalidArgument(String),
    /// Two linked provider registrations used the same provider kind.
    #[error("duplicate provider registration for kind `{kind}`")]
    DuplicateProviderRegistration {
        /// Duplicated provider kind.
        kind: String,
    },
    /// Configuration referenced a provider kind that no linked provider registered.
    #[error("unknown provider kind `{kind}`")]
    UnknownProviderKind {
        /// Provider kind from configuration.
        kind: ProviderKind,
    },
    /// Configuration defined the same provider instance id more than once.
    #[error("duplicate provider instance `{id}`")]
    DuplicateProviderInstance {
        /// Duplicated provider instance id.
        id: ProviderId,
    },
    /// Configuration defined the same deployment id more than once.
    #[error("duplicate deployment `{id}`")]
    DuplicateDeployment {
        /// Duplicated deployment id.
        id: DeploymentId,
    },
    /// A request model could not be resolved to a configured deployment.
    #[error("no deployment for model `{model}`")]
    NoDeploymentForModel {
        /// Requested model selector.
        model: ModelRef,
    },
    /// The selected provider instance does not expose chat capability.
    #[error("provider `{provider}` does not support chat")]
    ProviderDoesNotSupportChat {
        /// Provider instance selected for the request.
        provider: ProviderId,
    },
    /// Request contained OpenAI-compatible parameters unsupported by the provider.
    #[error("provider `{provider}` does not support params: {}", params.join(", "))]
    UnsupportedParams {
        /// Provider instance selected for the request.
        provider: ProviderId,
        /// Unsupported parameter names.
        params: Vec<String>,
    },
    /// Provider configuration was invalid.
    #[error("provider config error: {message}")]
    ProviderConfig {
        /// Provider instance related to the error, when known.
        provider: Option<ProviderId>,
        /// Human-readable configuration error.
        message: String,
    },
    /// Provider adapter failed while transforming request data.
    #[error("provider `{provider}` transform error: {message}")]
    ProviderTransform {
        /// Provider instance selected for the request.
        provider: ProviderId,
        /// Human-readable transform error.
        message: String,
    },
    /// Provider adapter failed while signing or authenticating a request.
    #[error("provider `{provider}` signing error: {message}")]
    ProviderSigning {
        /// Provider instance selected for the request.
        provider: ProviderId,
        /// Human-readable signing error.
        message: String,
    },
    /// HTTP request construction, sending, body collection, or streaming failed.
    #[error("http error: {message}")]
    Http {
        /// Human-readable HTTP client error.
        message: String,
    },
    /// Provider returned a non-success HTTP status with a business error body.
    ///
    /// This is the catch-all variant: adapters return it whenever they cannot
    /// classify the failure into one of the semantic variants below. Callers
    /// that want to react to specific failure modes should match on the
    /// dedicated variants first and fall back to this one for everything else.
    #[error("provider `{provider}` business error ({status}): {message}")]
    ProviderBusiness {
        /// Provider instance selected for the request.
        provider: ProviderId,
        /// HTTP status returned by the provider.
        status: StatusCode,
        /// Provider-native stable error code, when the provider returned one.
        code: Option<String>,
        /// Human-readable provider error message.
        message: String,
        /// Provider-native structured error details for callers that need
        /// provider-specific handling.
        details: Option<Value>,
    },
    /// Provider rejected the request with a rate limit or overload signal.
    ///
    /// Callers should typically back off and retry. `retry_after` carries the
    /// upstream `Retry-After` header value when the provider sent one (in
    /// seconds).
    #[error("provider `{provider}` rate limited ({status}): {message}")]
    RateLimited {
        /// Provider instance selected for the request.
        provider: ProviderId,
        /// HTTP status returned by the provider.
        status: StatusCode,
        /// Provider-native stable error code, when the provider returned one.
        code: Option<String>,
        /// Human-readable provider error message.
        message: String,
        /// Suggested retry delay parsed from the provider response, in seconds.
        retry_after: Option<u64>,
        /// Provider-native structured error details.
        details: Option<Value>,
    },
    /// Request exceeded the model's context window.
    ///
    /// Callers should shorten the prompt, switch to a larger model, or summarize
    /// the conversation before retrying.
    #[error("provider `{provider}` context window exceeded ({status}): {message}")]
    ContextWindowExceeded {
        /// Provider instance selected for the request.
        provider: ProviderId,
        /// HTTP status returned by the provider.
        status: StatusCode,
        /// Provider-native stable error code, when the provider returned one.
        code: Option<String>,
        /// Human-readable provider error message.
        message: String,
        /// Provider-native structured error details.
        details: Option<Value>,
    },
    /// Provider's safety system blocked the request or response.
    ///
    /// Retrying without changing the prompt will fail again. Callers usually
    /// surface this to end users so they can revise their input.
    #[error("provider `{provider}` content filtered ({status}): {message}")]
    ContentFiltered {
        /// Provider instance selected for the request.
        provider: ProviderId,
        /// HTTP status returned by the provider.
        status: StatusCode,
        /// Provider-native stable error code, when the provider returned one.
        code: Option<String>,
        /// Human-readable provider error message.
        message: String,
        /// Provider-native structured error details.
        details: Option<Value>,
    },
    /// Authentication or authorization failure.
    ///
    /// Callers should treat this as a configuration problem (missing or invalid
    /// credentials, revoked permissions). Retrying with the same credentials
    /// will fail again.
    #[error("provider `{provider}` auth failed ({status}): {message}")]
    AuthFailed {
        /// Provider instance selected for the request.
        provider: ProviderId,
        /// HTTP status returned by the provider.
        status: StatusCode,
        /// Provider-native stable error code, when the provider returned one.
        code: Option<String>,
        /// Human-readable provider error message.
        message: String,
        /// Provider-native structured error details.
        details: Option<Value>,
    },
    /// Provider response could not be transformed into sigma response types.
    #[error("provider `{provider}` response error: {message}")]
    ProviderResponse {
        /// Provider instance selected for the request.
        provider: ProviderId,
        /// Human-readable response error.
        message: String,
    },
}

impl From<derive_builder::UninitializedFieldError> for SigmaError {
    fn from(value: derive_builder::UninitializedFieldError) -> Self {
        Self::InvalidArgument(value.to_string())
    }
}

impl From<String> for SigmaError {
    fn from(value: String) -> Self {
        Self::InvalidArgument(value)
    }
}
