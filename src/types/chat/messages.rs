use std::fmt::Display;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::chat::content::{AudioPart, FilePart, ImagePart, RefusalPart, TextPart};
use crate::types::chat::tools::ToolCall;

/// Semantic role for a chat message.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Instructions supplied by the platform or application.
    System,
    /// User input.
    #[default]
    User,
    /// Model output.
    Assistant,
    /// Tool result returned to the model.
    Tool,
    /// Developer instruction with lower precedence than system instructions.
    Developer,
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Developer => "developer",
        };
        f.write_str(value)
    }
}

/// Provider-neutral reasoning data associated with an assistant message or
/// tool call.
///
/// Providers translate these blocks to their native thinking, signature, or
/// thought metadata when replaying multi-turn context.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReasoningBlock {
    /// Visible reasoning text and optional replay signature.
    Text {
        /// Reasoning text.
        text: String,
        /// Optional provider replay signature.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Redacted reasoning payload and optional replay signature.
    Redacted {
        /// Opaque redacted data returned by a provider.
        data: String,
        /// Optional provider replay signature.
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Opaque reasoning or thought signature without visible text.
    Signature {
        /// Signature value to replay with future messages or tool calls.
        value: String,
    },
}

impl ReasoningBlock {
    /// Creates a visible reasoning text block.
    pub fn text(text: impl Into<String>, signature: Option<impl Into<String>>) -> Self {
        Self::Text {
            text: text.into(),
            signature: signature.map(Into::into),
        }
    }

    /// Creates a redacted reasoning block.
    pub fn redacted(data: impl Into<String>, signature: Option<impl Into<String>>) -> Self {
        Self::Redacted {
            data: data.into(),
            signature: signature.map(Into::into),
        }
    }

    /// Creates a standalone reasoning signature block.
    pub fn signature(value: impl Into<String>) -> Self {
        Self::Signature {
            value: value.into(),
        }
    }

    /// Returns the visible reasoning text when this block contains one.
    pub fn text_value(&self) -> Option<&str> {
        match self {
            Self::Text { text, .. } => Some(text),
            Self::Redacted { .. } | Self::Signature { .. } => None,
        }
    }

    /// Returns a provider replay signature when this block carries one.
    pub fn signature_value(&self) -> Option<&str> {
        match self {
            Self::Text { signature, .. } | Self::Redacted { signature, .. } => signature.as_deref(),
            Self::Signature { value } => Some(value),
        }
    }
}

/// Provider-owned context that can be replayed only to the same provider.
///
/// This is the semantic chat layer's escape hatch for native context blocks
/// that are required for multi-turn correctness but are not portable model
/// content or reasoning. Providers that return hosted tool results,
/// compaction markers, or similar opaque replay state should preserve them
/// here and consume only entries whose [`ProviderContextBlock::provider`]
/// matches the active provider id.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ProviderContextBlock {
    /// Provider id that produced this context block.
    pub provider: String,
    /// Provider-defined context kind.
    ///
    /// The value is intentionally namespaced by convention, for example
    /// `anthropic.content_block` or `anthropic.response_field`.
    pub kind: String,
    /// Opaque provider-native context payload.
    pub value: Value,
}

impl ProviderContextBlock {
    /// Creates an opaque provider-context block.
    pub fn new(provider: impl Into<String>, kind: impl Into<String>, value: Value) -> Self {
        Self {
            provider: provider.into(),
            kind: kind.into(),
            value,
        }
    }
}

/// Text-only content for system and developer messages.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TextContent {
    /// Plain string content.
    Text(String),
    /// Structured text parts.
    Parts(Vec<TextPart>),
}

impl Default for TextContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<&str> for TextContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for TextContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// Content supplied by a user message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum UserContent {
    /// Plain text user content.
    Text(String),
    /// Structured multimodal user content.
    Parts(Vec<UserContentPart>),
}

impl Default for UserContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<&str> for UserContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for UserContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<Vec<UserContentPart>> for UserContent {
    fn from(value: Vec<UserContentPart>) -> Self {
        Self::Parts(value)
    }
}

/// Structured content part for a user message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContentPart {
    /// Text part.
    Text(TextPart),
    /// Image part.
    Image(ImagePart),
    /// Audio input part.
    Audio(AudioPart),
    /// File, document, image, or video file part.
    File(FilePart),
}

impl From<TextPart> for UserContentPart {
    fn from(value: TextPart) -> Self {
        Self::Text(value)
    }
}

impl From<ImagePart> for UserContentPart {
    fn from(value: ImagePart) -> Self {
        Self::Image(value)
    }
}

impl From<AudioPart> for UserContentPart {
    fn from(value: AudioPart) -> Self {
        Self::Audio(value)
    }
}

impl From<FilePart> for UserContentPart {
    fn from(value: FilePart) -> Self {
        Self::File(value)
    }
}

/// Content supplied by or replayed for an assistant message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AssistantContent {
    /// Plain assistant text.
    Text(String),
    /// Structured assistant content parts.
    Parts(Vec<AssistantContentPart>),
}

impl From<&str> for AssistantContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for AssistantContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// Structured content part for an assistant message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContentPart {
    /// Text content part.
    Text(TextPart),
    /// Refusal content part.
    Refusal(RefusalPart),
}

/// Content supplied by a tool result message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolContent {
    /// Plain tool result text.
    Text(String),
    /// Structured text parts for tool results.
    Parts(Vec<TextPart>),
}

impl Default for ToolContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl From<&str> for ToolContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<String> for ToolContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

/// Developer message carrying instructions for the model.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct DeveloperMessage {
    /// Developer instructions.
    pub content: TextContent,
    /// Optional caller-defined participant name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl DeveloperMessage {
    /// Creates a developer text message.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: TextContent::Text(content.into()),
            name: None,
        }
    }
}

/// System message carrying highest-priority instructions for the model.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct SystemMessage {
    /// System instructions.
    pub content: TextContent,
    /// Optional caller-defined participant name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl SystemMessage {
    /// Creates a system text message.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: TextContent::Text(content.into()),
            name: None,
        }
    }
}

/// User message carrying human or application input.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct UserMessage {
    /// User message content.
    pub content: UserContent,
    /// Optional caller-defined participant name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl UserMessage {
    /// Creates a user text message.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: UserContent::Text(content.into()),
            name: None,
        }
    }
}

impl From<&str> for UserMessage {
    fn from(value: &str) -> Self {
        Self::text(value)
    }
}

impl From<String> for UserMessage {
    fn from(value: String) -> Self {
        Self::text(value)
    }
}

/// Reference to audio previously returned by a model.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct AssistantAudio {
    /// Provider response audio identifier.
    pub id: String,
}

/// Assistant message returned by or replayed to a chat model.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct AssistantMessage {
    /// Assistant text or structured content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<AssistantContent>,
    /// Refusal text when the assistant declined to answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// Optional caller-defined participant name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Audio output reference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AssistantAudio>,
    /// Tool calls requested by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Provider-neutral reasoning data to display or replay.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Vec<ReasoningBlock>>,
    /// Opaque provider-owned context to replay only to the same provider.
    ///
    /// Applications may copy this from a [`crate::types::chat::ChatResponse`]
    /// assistant message into the next [`AssistantMessage`] when preserving a
    /// provider-native conversation. Providers must ignore entries they did not
    /// produce.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_context: Option<Vec<ProviderContextBlock>>,
}

impl AssistantMessage {
    /// Creates an assistant text message.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: Some(AssistantContent::Text(content.into())),
            ..Default::default()
        }
    }
}

impl From<&str> for AssistantMessage {
    fn from(value: &str) -> Self {
        Self::text(value)
    }
}

impl From<String> for AssistantMessage {
    fn from(value: String) -> Self {
        Self::text(value)
    }
}

/// Tool result message sent after an assistant tool call.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ToolMessage {
    /// Tool result content.
    pub content: ToolContent,
    /// Identifier of the assistant tool call this message satisfies.
    pub tool_call_id: String,
}

/// Provider-neutral chat message.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    /// Developer instructions.
    Developer(DeveloperMessage),
    /// System instructions.
    System(SystemMessage),
    /// User input.
    User(UserMessage),
    /// Assistant output or replay context.
    Assistant(AssistantMessage),
    /// Tool result.
    Tool(ToolMessage),
}

impl From<DeveloperMessage> for ChatMessage {
    fn from(value: DeveloperMessage) -> Self {
        Self::Developer(value)
    }
}

impl From<SystemMessage> for ChatMessage {
    fn from(value: SystemMessage) -> Self {
        Self::System(value)
    }
}

impl From<UserMessage> for ChatMessage {
    fn from(value: UserMessage) -> Self {
        Self::User(value)
    }
}

impl From<AssistantMessage> for ChatMessage {
    fn from(value: AssistantMessage) -> Self {
        Self::Assistant(value)
    }
}

impl From<ToolMessage> for ChatMessage {
    fn from(value: ToolMessage) -> Self {
        Self::Tool(value)
    }
}
