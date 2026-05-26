use schemars::JsonSchema;
use serde::Deserialize;

use crate::config::SecretString;
use crate::providers::common::non_empty_env;
use crate::{ModelName, ProviderInit, SigmaError, SigmaResult};

use super::BEDROCK_DEFAULT_REGION;

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(super) struct BedrockConfig {
    /// AWS region used for Bedrock Runtime endpoint selection and SigV4 scope.
    pub(super) region: Option<String>,
    /// Static AWS access key id for SigV4 request signing.
    pub(super) access_key_id: Option<SecretString>,
    /// Static AWS secret access key for SigV4 request signing.
    pub(super) secret_access_key: Option<SecretString>,
    /// Optional AWS session token for temporary credentials.
    pub(super) session_token: Option<SecretString>,
    /// Optional Bedrock bearer token. When present, sigma uses `Authorization: Bearer ...`.
    pub(super) bearer_token: Option<SecretString>,
    /// Optional Bedrock Runtime endpoint override.
    pub(super) runtime_endpoint: Option<String>,
}

#[derive(Debug, Clone)]
pub(super) struct AwsCredentials {
    pub(super) access_key_id: SecretString,
    pub(super) secret_access_key: SecretString,
    pub(super) session_token: Option<SecretString>,
}

#[derive(Debug, Clone)]
pub(super) enum BedrockAuth {
    Bearer(SecretString),
    SigV4(AwsCredentials),
    Header,
}

pub(super) fn resolve_configured_region(config_region: Option<&str>) -> Option<String> {
    config_region
        .map(str::to_string)
        .or_else(|| non_empty_env("AWS_REGION_NAME"))
        .or_else(|| non_empty_env("AWS_REGION"))
        .or_else(|| non_empty_env("AWS_DEFAULT_REGION"))
}

pub(super) fn resolve_region(configured_region: Option<&str>, model: &ModelName) -> String {
    configured_region
        .map(str::to_string)
        .or_else(|| region_from_bedrock_arn(model.as_str()))
        .unwrap_or_else(|| BEDROCK_DEFAULT_REGION.to_string())
}

pub(super) fn resolve_api_base_override<TConfig>(
    init: &ProviderInit<TConfig>,
    runtime_endpoint: Option<&str>,
) -> Option<String> {
    init.common
        .api_base
        .clone()
        .or_else(|| runtime_endpoint.map(str::to_string))
        .or_else(|| non_empty_env("AWS_BEDROCK_RUNTIME_ENDPOINT"))
}

pub(super) fn resolve_auth(
    init: &ProviderInit<BedrockConfig>,
    configured_authorization_header: bool,
) -> SigmaResult<BedrockAuth> {
    if configured_authorization_header {
        return Ok(BedrockAuth::Header);
    }

    if let Some(bearer_token) = init
        .config
        .bearer_token
        .clone()
        .or_else(|| init.common.api_key.clone())
        .or_else(|| non_empty_env("AWS_BEARER_TOKEN_BEDROCK").map(SecretString::from))
    {
        return Ok(BedrockAuth::Bearer(bearer_token));
    }

    let access_key_id = init
        .config
        .access_key_id
        .clone()
        .or_else(|| non_empty_env("AWS_ACCESS_KEY_ID").map(SecretString::from));
    let secret_access_key = init
        .config
        .secret_access_key
        .clone()
        .or_else(|| non_empty_env("AWS_SECRET_ACCESS_KEY").map(SecretString::from));
    let session_token = init
        .config
        .session_token
        .clone()
        .or_else(|| non_empty_env("AWS_SESSION_TOKEN").map(SecretString::from));

    let (Some(access_key_id), Some(secret_access_key)) = (access_key_id, secret_access_key) else {
        return Err(SigmaError::ProviderConfig {
            provider: Some(init.id.clone()),
            message: "bedrock provider requires bearer_token, AWS_BEARER_TOKEN_BEDROCK, Authorization header, or AWS SigV4 credentials via access_key_id/secret_access_key or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY".to_string(),
        });
    };

    Ok(BedrockAuth::SigV4(AwsCredentials {
        access_key_id,
        secret_access_key,
        session_token,
    }))
}

pub(super) fn region_from_bedrock_arn(model: &str) -> Option<String> {
    let parts = model.split(':').collect::<Vec<_>>();
    if parts.len() >= 4 && parts[0] == "arn" && parts[2].starts_with("bedrock") {
        let region = parts[3];
        if valid_aws_region(region) {
            return Some(region.to_string());
        }
    }
    None
}

fn valid_aws_region(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}
