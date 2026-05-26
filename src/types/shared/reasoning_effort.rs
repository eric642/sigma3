use serde::{Deserialize, Serialize};

/// Portable reasoning-effort hint for providers that support adjustable
/// reasoning or thinking budgets.
///
/// Providers map these values to their native controls when supported. Use
/// [`crate::types::chat::ChatRequest::with_provider_option`] for
/// provider-native reasoning controls that are not represented by this
/// portable scale.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// Disable provider reasoning when the selected provider supports it.
    None,
    /// Request the smallest available reasoning budget.
    Minimal,
    /// Request a low reasoning budget.
    Low,
    /// Request the provider's default balanced reasoning budget.
    #[default]
    Medium,
    /// Request a high reasoning budget.
    High,
    /// Request an extra-high reasoning budget.
    Xhigh,
    /// Request the maximum reasoning budget sigma exposes portably.
    Max,
}
