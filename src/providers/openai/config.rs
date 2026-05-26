use schemars::JsonSchema;
use serde::Deserialize;

use crate::config::SecretString;
use crate::providers::common::non_empty_env;
use crate::{ProviderInit, SigmaError, SigmaResult};

use super::OPENAI_DEFAULT_BASE_URL;

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(super) struct OpenAiConfig {}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(super) struct OpenAiCompatibleConfig {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OpenAiFlavor {
    OpenAi,
    Compatible,
}

impl OpenAiFlavor {
    pub(super) fn requires_authentication(self) -> bool {
        self == Self::OpenAi
    }

    pub(super) fn sanitizes_usage(self) -> bool {
        matches!(self, Self::Compatible)
    }
}

pub(super) fn resolve_api_base<TConfig>(
    init: &ProviderInit<TConfig>,
    flavor: OpenAiFlavor,
) -> SigmaResult<String> {
    match flavor {
        OpenAiFlavor::OpenAi => Ok(init
            .common
            .api_base
            .clone()
            .or_else(|| non_empty_env("OPENAI_BASE_URL"))
            .or_else(|| non_empty_env("OPENAI_API_BASE"))
            .unwrap_or_else(|| OPENAI_DEFAULT_BASE_URL.to_string())),
        OpenAiFlavor::Compatible => init
            .common
            .api_base
            .clone()
            .or_else(|| non_empty_env("OPENAI_COMPATIBLE_API_BASE"))
            .or_else(|| non_empty_env("OPENAI_LIKE_API_BASE"))
            .ok_or_else(|| SigmaError::ProviderConfig {
                provider: Some(init.id.clone()),
                message: "openai-compatible provider requires api_base, OPENAI_COMPATIBLE_API_BASE, or OPENAI_LIKE_API_BASE".to_string(),
            }),
    }
}

pub(super) fn resolve_api_key(
    api_key: Option<SecretString>,
    flavor: OpenAiFlavor,
) -> Option<SecretString> {
    api_key.or_else(|| match flavor {
        OpenAiFlavor::OpenAi => non_empty_env("OPENAI_API_KEY").map(SecretString::from),
        OpenAiFlavor::Compatible => non_empty_env("OPENAI_COMPATIBLE_API_KEY")
            .or_else(|| non_empty_env("OPENAI_LIKE_API_KEY"))
            .map(SecretString::from),
    })
}
