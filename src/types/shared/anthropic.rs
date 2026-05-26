use serde::{Deserialize, Serialize};

/// Anthropic extended thinking control sent to the Messages API.
///
/// Use this when callers need Claude's native `thinking` parameter instead of
/// the portable [`crate::types::shared::ReasoningEffort`] mapping. Providers
/// other than Anthropic should treat this as provider-specific metadata.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct AnthropicThinkingParam {
    /// Thinking mode requested from Anthropic.
    #[serde(rename = "type")]
    pub r#type: AnthropicThinkingType,
    /// Token budget for `enabled` thinking.
    ///
    /// Anthropic rejects values below its current minimum. When using
    /// [`crate::types::shared::ReasoningEffort`], sigma chooses the same
    /// default budgets LiteLLM uses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

/// Anthropic thinking mode.
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicThinkingType {
    /// Enable thinking with an explicit token budget.
    Enabled,
    /// Let Anthropic choose thinking behavior adaptively.
    Adaptive,
}

/// Anthropic thinking content returned by Claude.
///
/// These blocks are exposed separately from the assistant text so callers can
/// replay Anthropic conversations with thinking signatures intact.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct AnthropicThinkingBlock {
    /// Native Anthropic block type, such as `thinking` or `redacted_thinking`.
    #[serde(rename = "type")]
    pub r#type: String,
    /// Visible thinking text when Anthropic returned it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Anthropic thinking signature used for future conversation turns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Redacted thinking payload when Anthropic hides the reasoning text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

/// Anthropic server-side tool usage counts.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, Default)]
pub struct AnthropicServerToolUse {
    /// Number of hosted web search requests charged by Anthropic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_requests: Option<u32>,
    /// Number of Anthropic tool-search requests charged by Anthropic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_search_requests: Option<u32>,
}
