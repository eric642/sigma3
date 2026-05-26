use http::StatusCode;
use serde_json::Value;

use crate::provider_http::ProviderResponse;
use crate::providers::common::fallback_error_message;
use crate::{ChatAdapterContext, SigmaError};

pub(super) fn anthropic_error_response(
    context: &ChatAdapterContext<'_>,
    response: ProviderResponse,
) -> SigmaError {
    let body = serde_json::from_slice::<Value>(&response.body).ok();
    match body {
        Some(body) => error_from_body(context, response.status, body),
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
    SigmaError::ProviderBusiness {
        provider: context.provider.to_owned(),
        status,
        code,
        message,
        details: error.cloned().or(Some(body)),
    }
}
