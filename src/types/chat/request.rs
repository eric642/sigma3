use std::collections::HashMap;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::config::ProviderMetadataMap;
use crate::error::SigmaError;
use crate::model::ModelRef;
use crate::types::chat::messages::ChatCompletionRequestMessage;
use crate::types::chat::options::{
    ChatCompletionAudio, ChatCompletionStreamOptions, PredictionContent, PromptCacheRetention,
    ResponseModalities, ServiceTier, Verbosity, WebSearchOptions,
};
use crate::types::chat::tools::{ChatCompletionToolChoiceOption, ChatCompletionTools};
use crate::types::shared::{ReasoningEffort, ResponseFormat};

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Prompt {
    String(String),
    StringArray(Vec<String>),
    IntegerArray(Vec<u32>),
    ArrayOfIntegerArray(Vec<Vec<u32>>),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum StopConfiguration {
    String(String),
    StringArray(Vec<String>),
}

#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "CreateChatCompletionRequestArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "SigmaError"))]
pub struct CreateChatCompletionRequest {
    /// Messages that form the conversation sent to the model.
    pub messages: Vec<ChatCompletionRequestMessage>,
    /// Model selector used by sigma routing.
    ///
    /// Plain strings deserialize as [`ModelRef::model`], preserving
    /// OpenAI-compatible JSON while allowing callers to use deployment or
    /// provider-model routing in Rust.
    pub model: ModelRef,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<ResponseModalities>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopConfiguration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, i8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictionContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTools>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    /// Provider-scoped final request body overrides.
    ///
    /// The map key is a configured provider instance id, not a provider kind.
    /// For example, a Zhipu AI endpoint configured with
    /// `kind = "openai-compatible"` should use the provider id such as
    /// `"zhipu"` here. When that provider is selected, its adapter
    /// shallow-merges the matching object into the final provider request body
    /// after parameter mapping and adapter-generated fields.
    ///
    /// Values in this map have the highest request-body priority and may
    /// override generated fields such as `"model"`, `"messages"`, and
    /// `"stream"`. To send a provider-native OpenAI-style `metadata` field,
    /// include a `"metadata"` entry inside the provider's override object.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: ProviderMetadataMap,
}
