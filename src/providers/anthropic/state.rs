use std::collections::HashMap;

use crate::ChatAdapterContext;

/// Request-scoped state shared between the Anthropic adapter's request and
/// response/stream transforms.
///
/// Stored as a type-erased `Arc<dyn Any + Send + Sync>` in
/// [`ChatAdapterContext::provider_state`] and recovered with
/// [`ChatAdapterContext::provider_state_as`].
#[derive(Debug, Default, Clone)]
pub(super) struct AnthropicState {
    /// Sanitized-to-original tool name map used to restore caller-visible names
    /// in `tool_use` blocks before sigma builds the public response.
    pub(super) reverse_tool_map: HashMap<String, String>,
    /// Set when the request injected the synthetic `json_tool_call` tool to
    /// emulate `response_format` for models that lack native structured output.
    /// Response transforms recognize this flag to restore the tool's `input`
    /// payload as message content instead of returning it as a tool call.
    pub(super) response_format_fallback: bool,
}

pub(super) fn reverse_tool_map(context: &ChatAdapterContext<'_>) -> HashMap<String, String> {
    context
        .provider_state_as::<AnthropicState>()
        .map(|state| state.reverse_tool_map.clone())
        .unwrap_or_default()
}

pub(super) fn response_format_fallback(context: &ChatAdapterContext<'_>) -> bool {
    context
        .provider_state_as::<AnthropicState>()
        .map(|state| state.response_format_fallback)
        .unwrap_or(false)
}
