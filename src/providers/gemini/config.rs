use schemars::JsonSchema;
use serde::Deserialize;

use crate::ModelName;

use super::helpers::is_gemini_3_or_newer;

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(super) struct GeminiConfig {
    /// Gemini REST API version selection.
    pub(super) api_version: GeminiApiVersion,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(super) enum GeminiApiVersion {
    /// Use v1alpha for Gemini 3 models and v1beta for other Gemini chat models.
    #[default]
    Auto,
    /// Always use the v1 API path.
    V1,
    /// Always use the v1beta API path.
    V1Beta,
    /// Always use the v1alpha API path.
    V1Alpha,
}

impl GeminiApiVersion {
    pub(super) fn segment(self, model: &ModelName) -> &'static str {
        match self {
            Self::Auto if is_gemini_3_or_newer(model.as_str()) => "v1alpha",
            Self::Auto => "v1beta",
            Self::V1 => "v1",
            Self::V1Beta => "v1beta",
            Self::V1Alpha => "v1alpha",
        }
    }
}
