use serde::{Deserialize, Serialize};

use crate::types::chat::messages::{ProviderContextBlock, ReasoningBlock, Role};
use crate::types::chat::options::ServiceTier;
use crate::types::chat::tools::ToolCall;
use crate::types::shared::{CompletionTokensDetails, PromptTokensDetails};

/// Token usage statistics for a chat request.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
pub struct Usage {
    /// Prompt/input token count.
    pub prompt_tokens: u32,
    /// Completion/output token count.
    pub completion_tokens: u32,
    /// Total token count.
    pub total_tokens: u32,
    /// Tokens used to create a prompt-cache entry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// Tokens read from prompt cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    /// Hosted tool usage counters when a provider reports them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hosted_tool_use: Option<HostedToolUsage>,
    /// Inference geography when reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<String>,
    /// Provider speed tier when reported or inferred.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<String>,
    /// Prompt-token breakdown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// Completion-token breakdown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Hosted tool usage counters.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq, Default)]
pub struct HostedToolUsage {
    /// Number of hosted web search requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_requests: Option<u32>,
    /// Number of hosted tool-search requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_search_requests: Option<u32>,
}

/// Annotation attached to response text.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// URL citation for generated text.
    UrlCitation {
        /// Citation details.
        url_citation: UrlCitation,
    },
}

/// URL citation details.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
pub struct UrlCitation {
    /// End character index of the citation span.
    pub end_index: u32,
    /// Start character index of the citation span.
    pub start_index: u32,
    /// Title of the cited resource.
    pub title: String,
    /// URL of the cited resource.
    pub url: String,
}

/// Audio data returned with an assistant response.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
pub struct ResponseAudio {
    /// Provider audio identifier.
    pub id: String,
    /// Expiration timestamp.
    pub expires_at: u64,
    /// Base64-encoded audio data.
    pub data: String,
    /// Text transcript of the audio.
    pub transcript: String,
}

/// Assistant message returned by a provider.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChatResponseMessage {
    /// Final assistant content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Provider-neutral reasoning blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Vec<ReasoningBlock>>,
    /// Refusal text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// Tool calls requested by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Response annotations such as URL citations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<Annotation>>,
    /// Message role, normally [`Role::Assistant`].
    pub role: Role,
    /// Audio output payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ResponseAudio>,
    /// Opaque provider-owned context required for same-provider replay.
    ///
    /// This is not portable model content. Preserve it only when continuing a
    /// conversation with the same provider that produced the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_context: Option<Vec<ProviderContextBlock>>,
}

/// Reason a model stopped generating.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// A natural stop condition or stop sequence was reached.
    Stop,
    /// Token or length limit was reached.
    Length,
    /// The model produced tool calls.
    ToolCalls,
    /// Provider content filters stopped the response.
    ContentFilter,
}

/// Top log probability alternative for a token.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct TokenTopLogprob {
    /// Token text.
    pub token: String,
    /// Log probability.
    pub logprob: f32,
    /// Token bytes when provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Log probability data for one generated token.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct TokenLogprob {
    /// Token text.
    pub token: String,
    /// Token log probability.
    pub logprob: f32,
    /// Token bytes when provided.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    /// Alternative token probabilities.
    pub top_logprobs: Vec<TokenTopLogprob>,
}

/// Log probability data for a chat choice.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChoiceLogprobs {
    /// Content token log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TokenLogprob>>,
    /// Refusal token log probabilities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Vec<TokenLogprob>>,
}

/// One choice in a chat response.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChatChoice {
    /// Choice index.
    pub index: u32,
    /// Assistant message.
    pub message: ChatResponseMessage,
    /// Finish reason when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// Token log probability details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Provider-neutral chat response returned by sigma.
#[derive(Debug, Deserialize, Clone, PartialEq, Serialize)]
pub struct ChatResponse {
    /// Provider response id.
    pub id: String,
    /// Generated choices.
    pub choices: Vec<ChatChoice>,
    /// Unix timestamp supplied by the provider or synthesized by sigma.
    pub created: u32,
    /// Provider model that handled the request.
    pub model: String,
    /// Service tier used by the provider when reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Response object type.
    pub object: String,
    /// Token usage details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}
