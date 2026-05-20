use serde::{Deserialize, Serialize};

use crate::types::chat::content::ChatCompletionRequestMessageContentPartText;

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

/// The retention policy for the prompt cache.
///
/// For most models the default is `in_memory`. For `gpt-5.5`, `gpt-5.5-pro`,
/// and all future models, the default is `24h` and `in_memory` is not supported.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub enum PromptCacheRetention {
    #[serde(rename = "in_memory")]
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

/// Constrains the verbosity of the model's response.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    Low,
    #[default]
    Medium,
    High,
}

/// Output types that you would like the model to generate for this request.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ResponseModalities {
    Text,
    Audio,
}

/// The amount of context window space to use for the search.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchContextSize {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchUserLocationType {
    Approximate,
}

/// Approximate location parameters for the search.
#[derive(Clone, Serialize, Debug, Default, Deserialize, PartialEq)]
pub struct WebSearchLocation {
    pub country: Option<String>,
    pub region: Option<String>,
    pub city: Option<String>,
    pub timezone: Option<String>,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct WebSearchUserLocation {
    pub r#type: WebSearchUserLocationType,
    pub approximate: WebSearchLocation,
}

/// Options for the web search tool.
#[derive(Clone, Serialize, Debug, Default, Deserialize, PartialEq)]
pub struct WebSearchOptions {
    pub search_context_size: Option<WebSearchContextSize>,
    pub user_location: Option<WebSearchUserLocation>,
}

/// The content that should be matched when generating a model response.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PredictionContentContent {
    Text(String),
    Array(Vec<ChatCompletionRequestMessageContentPartText>),
}

/// Static predicted output content.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase", content = "content")]
pub enum PredictionContent {
    Content(PredictionContentContent),
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionAudioVoice {
    Alloy,
    Ash,
    Ballad,
    Coral,
    Echo,
    Fable,
    Nova,
    Onyx,
    Sage,
    Shimmer,
    #[serde(untagged)]
    Other(String),
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionAudioFormat {
    Wav,
    Aac,
    Mp3,
    Flac,
    Opus,
    Pcm16,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionAudio {
    pub voice: ChatCompletionAudioVoice,
    pub format: ChatCompletionAudioFormat,
}

/// Options for streaming response. Only set this when you set `stream: true`.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq)]
pub struct ChatCompletionStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}
