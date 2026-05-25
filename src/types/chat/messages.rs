use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::error::SigmaError;
use crate::types::chat::content::{
    ChatCompletionRequestMessageContentPartAudio, ChatCompletionRequestMessageContentPartFile,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartRefusal,
    ChatCompletionRequestMessageContentPartText,
};
use crate::types::chat::tools::ChatCompletionMessageToolCalls;
use crate::types::shared::AnthropicThinkingBlock;

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionRequestDeveloperMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionRequestSystemMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionRequestUserMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
    ImageUrl(ChatCompletionRequestMessageContentPartImage),
    InputAudio(ChatCompletionRequestMessageContentPartAudio),
    File(ChatCompletionRequestMessageContentPartFile),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionRequestAssistantMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
    Refusal(ChatCompletionRequestMessageContentPartRefusal),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionRequestToolMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestDeveloperMessageContent {
    Text(String),
    Array(Vec<ChatCompletionRequestDeveloperMessageContentPart>),
}

impl Default for ChatCompletionRequestDeveloperMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestSystemMessageContent {
    Text(String),
    Array(Vec<ChatCompletionRequestSystemMessageContentPart>),
}

impl Default for ChatCompletionRequestSystemMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestUserMessageContent {
    Text(String),
    Array(Vec<ChatCompletionRequestUserMessageContentPart>),
}

impl Default for ChatCompletionRequestUserMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestAssistantMessageContent {
    Text(String),
    Array(Vec<ChatCompletionRequestAssistantMessageContentPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestToolMessageContent {
    Text(String),
    Array(Vec<ChatCompletionRequestToolMessageContentPart>),
}

impl Default for ChatCompletionRequestToolMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestDeveloperMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "SigmaError"))]
pub struct ChatCompletionRequestDeveloperMessage {
    pub content: ChatCompletionRequestDeveloperMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestSystemMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "SigmaError"))]
pub struct ChatCompletionRequestSystemMessage {
    pub content: ChatCompletionRequestSystemMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestUserMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "SigmaError"))]
pub struct ChatCompletionRequestUserMessage {
    pub content: ChatCompletionRequestUserMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestAssistantMessageAudio {
    /// Unique identifier for a previous audio response from the model.
    pub id: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestAssistantMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "SigmaError"))]
pub struct ChatCompletionRequestAssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatCompletionRequestAssistantMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionRequestAssistantMessageAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCalls>>,
    /// Anthropic thinking blocks from a previous Claude response.
    ///
    /// Preserve these when replaying Anthropic conversation history so Claude
    /// can verify thinking signatures on later turns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_blocks: Option<Vec<AnthropicThinkingBlock>>,
    /// Provider-native fields needed to replay Anthropic server tool and
    /// context-management content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_specific_fields: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Builder, PartialEq)]
#[builder(name = "ChatCompletionRequestToolMessageArgs")]
#[builder(pattern = "mutable")]
#[builder(setter(into, strip_option), default)]
#[builder(derive(Debug))]
#[builder(build_fn(error = "SigmaError"))]
pub struct ChatCompletionRequestToolMessage {
    pub content: ChatCompletionRequestToolMessageContent,
    pub tool_call_id: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatCompletionRequestMessage {
    Developer(ChatCompletionRequestDeveloperMessage),
    System(ChatCompletionRequestSystemMessage),
    User(ChatCompletionRequestUserMessage),
    Assistant(ChatCompletionRequestAssistantMessage),
    Tool(ChatCompletionRequestToolMessage),
}
