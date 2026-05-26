use serde::{Deserialize, Serialize};

use crate::types::chat::content::TextPart;

/// Provider service tier requested for a chat call.
#[derive(Clone, Copy, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    /// Let the provider choose the tier.
    Auto,
    /// Provider default tier.
    Default,
    /// Lower-latency flexible tier.
    Flex,
    /// Scale tier.
    Scale,
    /// Priority tier.
    Priority,
}

/// Retention policy for prompt-cache entries.
#[derive(Clone, Copy, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheRetention {
    /// Keep cache entries only in memory when supported.
    InMemory,
    /// Keep cache entries for twenty-four hours when supported.
    #[serde(rename = "24h")]
    TwentyFourHours,
}

/// Controls the verbosity of model-generated text.
#[derive(Clone, Copy, Serialize, Debug, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    /// Prefer concise output.
    Low,
    /// Use provider default verbosity.
    #[default]
    Medium,
    /// Prefer more detailed output.
    High,
}

/// Output modality requested from a multimodal model.
#[derive(Clone, Copy, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OutputModality {
    /// Text output.
    Text,
    /// Audio output.
    Audio,
}

/// Amount of model context available to hosted web search.
#[derive(Clone, Copy, Serialize, Debug, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchContextSize {
    /// Small search context.
    Low,
    /// Provider default search context.
    #[default]
    Medium,
    /// Larger search context.
    High,
}

/// User-location approximation mode for hosted web search.
#[derive(Clone, Copy, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchUserLocationType {
    /// Approximate user location.
    Approximate,
}

/// Approximate location parameters for hosted web search.
#[derive(Clone, Serialize, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct WebSearchLocation {
    /// ISO country code when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    /// Region or state when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// City when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// IANA timezone when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

/// User location hint for hosted web search.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
pub struct WebSearchUserLocation {
    /// Location hint type.
    pub r#type: WebSearchUserLocationType,
    /// Approximate location details.
    pub approximate: WebSearchLocation,
}

/// Hosted web-search options.
#[derive(Clone, Serialize, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct WebSearchOptions {
    /// Search-context budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<WebSearchContextSize>,
    /// Optional user-location hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<WebSearchUserLocation>,
}

/// Static content supplied as a prediction hint.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PredictionContentValue {
    /// Plain predicted text.
    Text(String),
    /// Structured predicted text parts.
    Parts(Vec<TextPart>),
}

/// Prediction hint used by providers that support static predicted output.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase", content = "content")]
pub enum PredictionContent {
    /// Static content prediction.
    Content(PredictionContentValue),
}

/// Voice requested for audio output.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AudioVoice {
    /// Alloy voice.
    Alloy,
    /// Ash voice.
    Ash,
    /// Ballad voice.
    Ballad,
    /// Coral voice.
    Coral,
    /// Echo voice.
    Echo,
    /// Fable voice.
    Fable,
    /// Nova voice.
    Nova,
    /// Onyx voice.
    Onyx,
    /// Sage voice.
    Sage,
    /// Shimmer voice.
    Shimmer,
    /// Provider-specific voice name.
    #[serde(untagged)]
    Other(String),
}

/// Audio output encoding format.
#[derive(Clone, Copy, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AudioOutputFormat {
    /// WAV audio.
    Wav,
    /// AAC audio.
    Aac,
    /// MP3 audio.
    Mp3,
    /// FLAC audio.
    Flac,
    /// Opus audio.
    Opus,
    /// 16-bit PCM audio.
    Pcm16,
}

/// Audio output configuration.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
pub struct AudioOutput {
    /// Voice requested from the provider.
    pub voice: AudioVoice,
    /// Encoding format for generated audio.
    pub format: AudioOutputFormat,
}

/// Options for streamed chat responses.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamOptions {
    /// Whether to include usage in the final stream chunk when supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
    /// Whether to include provider obfuscation markers when supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}
