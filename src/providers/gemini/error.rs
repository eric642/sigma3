use serde_json::Value;

use crate::provider_http::ProviderResponse;
use crate::providers::common::{
    classify_provider_error, fallback_error_message, parse_retry_after,
};
use crate::{ChatAdapterContext, SigmaError};

pub(super) fn gemini_error_response(
    context: &ChatAdapterContext<'_>,
    response: ProviderResponse,
) -> SigmaError {
    let body = serde_json::from_slice::<Value>(&response.body).ok();
    let error = body
        .as_ref()
        .and_then(|body| body.get("error"))
        .and_then(Value::as_object);
    let code = error
        .and_then(|error| error.get("status"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            error
                .and_then(|error| error.get("code"))
                .map(|code| code.to_string())
        });
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| fallback_error_message(response.status, &response.body));
    let retry_after = parse_retry_after(&response.headers);

    if let Some(err) = classify_provider_error(
        context.provider,
        response.status,
        code.as_deref(),
        &message,
        retry_after,
        body.clone(),
    ) {
        return err;
    }

    SigmaError::ProviderBusiness {
        provider: context.provider.to_owned(),
        status: response.status,
        code,
        message,
        details: body,
    }
}
