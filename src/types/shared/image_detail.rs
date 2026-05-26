use serde::{Deserialize, Serialize};

/// Requested media detail level for image, document, or video inputs.
///
/// Providers interpret these hints according to their own model capabilities
/// and may ignore or reject unsupported detail levels.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Let the selected provider choose an appropriate media resolution.
    #[default]
    Auto,
    /// Prefer lower resolution media processing to reduce cost or latency.
    Low,
    /// Prefer medium resolution media processing when supported.
    Medium,
    /// Prefer high resolution media processing.
    High,
    /// Prefer the highest explicit media resolution offered by the provider.
    UltraHigh,
    /// Preserve original media detail when the provider supports it.
    Original,
}
