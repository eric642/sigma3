use http::StatusCode;
use serde_json::Value;

use crate::provider_http::ProviderResponse;
use crate::providers::common::{
    classify_provider_error, fallback_error_message, parse_retry_after,
};
use crate::{ChatAdapterContext, ProviderId, SigmaError};

pub(super) fn anthropic_error_response(
    context: &ChatAdapterContext<'_>,
    response: ProviderResponse,
) -> SigmaError {
    let body = serde_json::from_slice::<Value>(&response.body).ok();
    let retry_after = parse_retry_after(&response.headers);
    match body {
        Some(body) => error_from_body(context, response.status, body, retry_after),
        None => SigmaError::ProviderBusiness {
            provider: context.provider.to_owned(),
            status: response.status,
            code: None,
            message: fallback_error_message(response.status, &response.body),
            details: None,
        },
    }
}

pub(super) fn error_from_body(
    context: &ChatAdapterContext<'_>,
    status: StatusCode,
    body: Value,
    retry_after: Option<u64>,
) -> SigmaError {
    let error = body.get("error").filter(|error| error.is_object());
    let code = error
        .and_then(|error| error.get("type").or_else(|| error.get("code")))
        .and_then(Value::as_str)
        .map(str::to_string);
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| body.to_string());
    let details = error.cloned().or(Some(body));

    if let Some(err) = classify_provider_error(
        context.provider,
        status,
        code.as_deref(),
        &message,
        retry_after,
        details.clone(),
    ) {
        return err;
    }

    SigmaError::ProviderBusiness {
        provider: context.provider.to_owned(),
        status,
        code,
        message,
        details,
    }
}

/// Maps Anthropic SSE `error.type` values to HTTP-style status codes.
///
/// The Messages API does not surface a status with mid-stream errors, so the
/// adapter infers one from the error type. This keeps semantic classification
/// (rate limits, auth, content policy) consistent between request-time and
/// stream-time failures.
pub(super) fn error_type_to_status(error_type: &str) -> StatusCode {
    match error_type {
        "overloaded_error" => StatusCode::from_u16(529).unwrap_or(StatusCode::SERVICE_UNAVAILABLE),
        "rate_limit_error" => StatusCode::TOO_MANY_REQUESTS,
        "authentication_error" => StatusCode::UNAUTHORIZED,
        "permission_error" => StatusCode::FORBIDDEN,
        "not_found_error" => StatusCode::NOT_FOUND,
        "invalid_request_error" => StatusCode::BAD_REQUEST,
        "request_too_large" => StatusCode::PAYLOAD_TOO_LARGE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Builds a [`SigmaError`] from a streaming `error` event payload.
pub(super) fn stream_error_from_event(
    provider: &ProviderId,
    error_type: Option<&str>,
    message: &str,
    details: Option<Value>,
) -> SigmaError {
    let status = error_type
        .map(error_type_to_status)
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    if let Some(err) =
        classify_provider_error(provider, status, error_type, message, None, details.clone())
    {
        return err;
    }
    SigmaError::ProviderBusiness {
        provider: provider.clone(),
        status,
        code: error_type.map(str::to_string),
        message: message.to_string(),
        details,
    }
}
