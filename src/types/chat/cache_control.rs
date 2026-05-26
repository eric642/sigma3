use serde::{Deserialize, Serialize};

/// Cache-control configuration for prompt caching.
///
/// Use this on content blocks for explicit cache breakpoints, or on
/// [`crate::types::chat::ChatRequestParams::cache_control`] for
/// request-level cache behavior. Providers decide how to translate this
/// semantic control into their native request format, and may ignore it when
/// unsupported.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct CacheControl {
    /// Cache-control mode requested by the caller.
    #[serde(rename = "type")]
    pub r#type: CacheControlType,
    /// Optional cache lifetime.
    ///
    /// When omitted, providers should use their default ephemeral cache
    /// lifetime if they support one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CacheControlTtl>,
}

impl CacheControl {
    /// Creates an ephemeral cache-control value using the provider default
    /// time-to-live.
    pub fn ephemeral() -> Self {
        Self {
            r#type: CacheControlType::Ephemeral,
            ttl: None,
        }
    }

    /// Creates an ephemeral cache-control value with an explicit time-to-live.
    pub fn ephemeral_with_ttl(ttl: CacheControlTtl) -> Self {
        Self {
            r#type: CacheControlType::Ephemeral,
            ttl: Some(ttl),
        }
    }
}

impl Default for CacheControl {
    fn default() -> Self {
        Self::ephemeral()
    }
}

/// Cache-control mode.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CacheControlType {
    /// Mark the block or request as eligible for ephemeral prompt caching.
    #[default]
    Ephemeral,
}

/// Cache-control time-to-live.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum CacheControlTtl {
    /// Cache for five minutes.
    #[serde(rename = "5m")]
    FiveMinutes,
    /// Cache for one hour.
    #[serde(rename = "1h")]
    OneHour,
}
