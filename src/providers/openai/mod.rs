use std::sync::Arc;

use crate::providers::openai_compatible::{
    OpenAiCompatibleConfig, OpenAiCompatibleProvider, OpenAiCompatibleProviderSpec,
};
use crate::{ProviderDriver, ProviderInit, ProviderKindStatic, SigmaResult, submit_provider};

mod config;

use config::OpenAiConfig;

const OPENAI_KIND: ProviderKindStatic = ProviderKindStatic::new("openai");
const OPENAI_DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const OPENAI_SPEC: OpenAiCompatibleProviderSpec = OpenAiCompatibleProviderSpec {
    default_api_base: Some(OPENAI_DEFAULT_BASE_URL),
    api_base_env: &["OPENAI_BASE_URL", "OPENAI_API_BASE"],
    api_key_env: &["OPENAI_API_KEY"],
    requires_authentication: true,
    sanitize_null_usage_tokens: false,
};

fn from_config(init: ProviderInit<OpenAiConfig>) -> SigmaResult<Arc<dyn ProviderDriver>> {
    OpenAiCompatibleProvider::from_init(init, OpenAiCompatibleConfig::default(), OPENAI_SPEC)
}

submit_provider! {
    kind: OPENAI_KIND,
    constructor: from_config,
    config: OpenAiConfig,
}
