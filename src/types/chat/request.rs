use std::collections::HashMap;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::config::{ChatParameterMap, ProviderOptionsMap};
use crate::error::SigmaError;
use crate::model::ModelRef;
use crate::types::chat::messages::ChatCompletionRequestMessage;
use crate::types::chat::options::{
    ChatCompletionAudio, ChatCompletionStreamOptions, PredictionContent, PromptCacheRetention,
    ResponseModalities, ServiceTier, Verbosity, WebSearchOptions,
};
use crate::types::chat::tools::{ChatCompletionToolChoiceOption, ChatCompletionTools};
use crate::types::shared::{AnthropicThinkingParam, ReasoningEffort, ResponseFormat};

use super::cache_control::CacheControl;

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

    /// OpenAI-compatible chat completion parameters.
    ///
    /// These fields are flattened into the serialized request body so the wire
    /// format remains OpenAI-compatible while keeping routing fields separate
    /// from model parameters in Rust.
    #[serde(flatten)]
    pub params: CreateChatCompletionRequestParams,
    /// Provider-scoped request options.
    ///
    /// The map key is a configured provider instance id, not a provider kind.
    /// For example, a Zhipu AI endpoint configured with
    /// `kind = "openai-compatible"` should use the provider id such as
    /// `"zhipu"` here. When that provider is selected, standard adapters
    /// shallow-merge the matching object into the final provider request body
    /// after parameter mapping and adapter-generated fields.
    ///
    /// Values in this map have the highest request-body priority and may
    /// override generated fields such as `"model"`, `"messages"`, and
    /// `"stream"`. Provider adapters may also reserve option keys for headers
    /// or other provider-specific controls. To send a provider-native
    /// OpenAI-style `metadata` field, include a `"metadata"` entry inside the
    /// provider's options object.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub provider_options: ProviderOptionsMap,
}

/// OpenAI-compatible chat completion parameters.
///
/// This type contains request fields that are sent as provider parameters after
/// deployment defaults are applied and routing-only fields are removed. It is
/// flattened into [`CreateChatCompletionRequest`] during JSON serialization.
#[derive(Clone, Serialize, Default, Debug, Builder, Deserialize, PartialEq)]
#[builder(name = "CreateChatCompletionRequestParamsArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "SigmaError"))]
pub struct CreateChatCompletionRequestParams {
    /// Output modalities requested from multimodal-capable models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<ResponseModalities>>,
    /// Controls text verbosity for models that support verbosity tuning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
    /// Reasoning effort requested for reasoning-capable models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Maximum number of output tokens the provider may generate.
    ///
    /// Anthropic's Messages API requires `max_tokens`. OpenAI-compatible
    /// callers may use either this field or [`CreateChatCompletionRequestParams::max_completion_tokens`];
    /// the Anthropic provider maps both to native `max_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Maximum number of completion tokens the model may generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// Penalty applied to repeated token frequency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Penalty applied when tokens have already appeared in the context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Web search controls for providers that support hosted search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
    /// Number of top log probabilities to include when logprobs are enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    /// Requested response formatting mode or schema.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Audio output configuration for audio-capable models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionAudio>,
    /// Whether the provider should store the completion when supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Whether the request asks for provider streaming.
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
    pub n: Option<u8>,
    /// Predicted output content for providers that support prediction hints.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictionContent>,
    /// Options that control streamed responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    /// Provider service tier selection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Anthropic top-k sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Tools the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTools>>,
    /// Tool selection mode or named tool choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    /// Whether the model may issue multiple tool calls in parallel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// Stable safety identifier for provider abuse monitoring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Prompt cache key for providers that support prompt caching.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Prompt cache retention policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    /// Native Anthropic thinking controls.
    ///
    /// Use this for provider-specific control. For portable reasoning hints,
    /// prefer [`CreateChatCompletionRequestParams::reasoning_effort`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinkingParam>,
    /// Request-level cache-control configuration.
    ///
    /// Use content part cache-control fields for explicit cache breakpoints.
    /// Providers translate this semantic hint to their native request-level
    /// cache-control shape when supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

impl CreateChatCompletionRequest {
    pub(crate) fn chat_parameters(&self) -> Result<ChatParameterMap, SigmaError> {
        match serde_json::to_value(&self.params)
            .map_err(|err| SigmaError::InvalidArgument(err.to_string()))?
        {
            serde_json::Value::Object(params) => Ok(params),
            _ => Err(SigmaError::InvalidArgument(
                "chat completion request parameters did not serialize to an object".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::config::ChatParameterMap;
    use crate::model::ProviderId;
    use crate::types::chat::messages::{
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    };

    #[test]
    fn chat_parameters_match_serialized_request_without_routing_fields() {
        let mut overrides = ChatParameterMap::new();
        overrides.insert("metadata".to_string(), json!({"trace_id": "trace-123"}));

        let request = CreateChatCompletionRequestArgs::default()
            .messages(vec![ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text("hi".to_string()),
                    name: None,
                },
            )])
            .model(ModelRef::model("gpt-public"))
            .params(
                CreateChatCompletionRequestParamsArgs::default()
                    .temperature(0.7)
                    .n(2)
                    .build()
                    .unwrap(),
            )
            .provider_options(HashMap::from([(ProviderId::from("selected"), overrides)]))
            .build()
            .unwrap();
        let mut expected = serde_json::to_value(&request)
            .unwrap()
            .as_object()
            .unwrap()
            .clone();
        expected.remove("messages");
        expected.remove("model");
        expected.remove("provider_options");

        let params = request.chat_parameters().unwrap();

        assert!(!params.contains_key("messages"));
        assert!(!params.contains_key("model"));
        assert!(!params.contains_key("provider_options"));
        assert_eq!(params, expected);
    }
}
