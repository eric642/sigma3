use std::collections::HashMap;
use std::env;

use http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use serde_json::Value;

use crate::{ProviderId, SigmaError, SigmaResult};

pub(super) fn non_empty_env(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.is_empty())
}

pub(super) fn header_map_from_config(
    provider: &ProviderId,
    headers: HashMap<String, String>,
) -> SigmaResult<HeaderMap> {
    let mut header_map = HeaderMap::new();

    for (name, value) in headers {
        let name =
            HeaderName::from_bytes(name.as_bytes()).map_err(|err| SigmaError::ProviderConfig {
                provider: Some(provider.clone()),
                message: format!("invalid header name `{name}`: {err}"),
            })?;
        let value = HeaderValue::from_str(&value).map_err(|err| SigmaError::ProviderConfig {
            provider: Some(provider.clone()),
            message: format!("invalid header value for `{name}`: {err}"),
        })?;
        header_map.insert(name, value);
    }

    Ok(header_map)
}

pub(super) fn signing_header_value(
    provider: &ProviderId,
    name: &str,
    value: &str,
) -> SigmaResult<HeaderValue> {
    HeaderValue::from_str(value).map_err(|err| SigmaError::ProviderSigning {
        provider: provider.clone(),
        message: format!("invalid header value for `{name}`: {err}"),
    })
}

pub(super) fn parse_response_json(provider: &ProviderId, body: &[u8]) -> SigmaResult<Value> {
    serde_json::from_slice(body).map_err(|err| SigmaError::ProviderResponse {
        provider: provider.clone(),
        message: err.to_string(),
    })
}

pub(super) fn fallback_error_message(status: StatusCode, body: &[u8]) -> String {
    if body.is_empty() {
        status
            .canonical_reason()
            .unwrap_or("provider returned unsuccessful HTTP status")
            .to_string()
    } else {
        String::from_utf8_lossy(body).into_owned()
    }
}

pub(super) fn event_data(
    event: &str,
    include_raw_json: bool,
    include_done: bool,
) -> Option<String> {
    let mut data_lines = Vec::new();

    for line in event.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        } else if (include_raw_json && line.starts_with('{')) || (include_done && line == "[DONE]")
        {
            data_lines.push(line.to_string());
        }
    }

    (!data_lines.is_empty()).then(|| data_lines.join("\n"))
}

pub(super) fn parse_data_uri(value: &str) -> Option<(&str, &str)> {
    let value = value.strip_prefix("data:")?;
    let (mime_type, data) = value.split_once(";base64,")?;
    Some((mime_type, data))
}
