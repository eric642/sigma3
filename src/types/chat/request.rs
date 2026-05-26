use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{ChatParameterMap, ProviderOptionsMap};
use crate::error::SigmaError;
use crate::model::{ModelRef, ProviderId};
use crate::types::chat::messages::ChatMessage;
use crate::types::chat::options::{
    AudioOutput, OutputModality, PredictionContent, PromptCacheRetention, ServiceTier,
    StreamOptions, Verbosity, WebSearchOptions,
};
use crate::types::chat::tools::{ToolChoice, ToolDefinition};
use crate::types::shared::{ReasoningEffort, ResponseFormat};

use super::cache_control::CacheControl;

/// Stop sequence configuration for a chat request.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub enum StopConfiguration {
    /// A single stop sequence.
    String(String),
    /// Multiple stop sequences.
    StringArray(Vec<String>),
}

/// Provider-neutral chat request.
///
/// `model` is resolved through sigma's deployment router. `params` contains
/// semantic chat controls that providers translate to native request fields.
/// `provider_options` is the explicit escape hatch for provider-native request
/// overrides and is keyed by configured provider id.
#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct ChatRequest {
    /// Conversation messages sent to the model.
    pub messages: Vec<ChatMessage>,
    /// Model selector used by sigma routing.
    pub model: ModelRef,
    /// Provider-neutral chat parameters.
    #[serde(default, skip_serializing_if = "ChatRequestParams::is_empty")]
    pub params: ChatRequestParams,
    /// Provider-scoped native request overrides.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub provider_options: ProviderOptionsMap,
}

impl ChatRequest {
    /// Creates a chat request for a routed model and message list.
    pub fn new(model: ModelRef, messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            model,
            params: ChatRequestParams::default(),
            provider_options: ProviderOptionsMap::default(),
        }
    }

    /// Returns this request with semantic parameters attached.
    pub fn with_params(mut self, params: ChatRequestParams) -> Self {
        self.params = params;
        self
    }

    /// Adds one provider-native option under the selected provider id.
    ///
    /// Provider options override generated provider request-body fields after
    /// semantic mapping. Use them only for provider-native behavior that sigma
    /// does not model directly.
    pub fn with_provider_option(
        mut self,
        provider: ProviderId,
        key: impl Into<String>,
        value: Value,
    ) -> Self {
        self.provider_options
            .entry(provider)
            .or_default()
            .insert(key.into(), value);
        self
    }

    pub(crate) fn chat_parameters(&self) -> Result<ChatParameterMap, SigmaError> {
        match serde_json::to_value(&self.params)
            .map_err(|err| SigmaError::InvalidArgument(err.to_string()))?
        {
            Value::Object(params) => Ok(params),
            _ => Err(SigmaError::InvalidArgument(
                "chat request parameters did not serialize to an object".to_string(),
            )),
        }
    }
}

/// Provider-neutral chat request parameters.
///
/// Providers map these semantic fields to native names and reject unsupported
/// parameters through their configured parameter policy.
#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct ChatRequestParams {
    /// Output modalities requested from multimodal models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_modalities: Option<Vec<OutputModality>>,
    /// Text verbosity hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
    /// Portable reasoning effort hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Maximum completion tokens for providers that distinguish completion
    /// tokens from other output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Frequency penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Hosted web-search options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search: Option<WebSearchOptions>,
    /// Number of top log probabilities to include when supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    /// Structured response format hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Audio output configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_output: Option<AudioOutput>,
    /// Whether the provider should store the completion when supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Whether this request should stream provider output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Stop sequence configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopConfiguration>,
    /// Token bias map keyed by provider tokenizer token id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, i8>>,
    /// Whether token log probabilities should be returned.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// Number of completions to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<u8>,
    /// Predicted output content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictionContent>,
    /// Stream response options.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// Provider service tier selection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Tool definitions available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool selection policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    /// Whether the model may issue tool calls in parallel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Stable safety identifier for provider abuse monitoring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Prompt cache key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Prompt cache retention policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    /// Request-level prompt-cache behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl ChatRequestParams {
    /// Returns true when no chat parameters are set.
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::chat::messages::UserMessage;

    #[test]
    fn chat_parameters_exclude_routing_fields() {
        let request = ChatRequest::new(
            ModelRef::model("model-public"),
            vec![UserMessage::text("hi").into()],
        )
        .with_params(ChatRequestParams {
            temperature: Some(0.7),
            count: Some(2),
            ..Default::default()
        })
        .with_provider_option(
            ProviderId::from("selected"),
            "metadata",
            json!({"trace": "x"}),
        );

        let params = request.chat_parameters().unwrap();

        assert_eq!(params.get("temperature"), Some(&json!(0.7f32)));
        assert_eq!(params.get("count"), Some(&json!(2)));
        assert!(!params.contains_key("messages"));
        assert!(!params.contains_key("model"));
        assert!(!params.contains_key("provider_options"));
    }
}
