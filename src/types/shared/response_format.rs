use serde::{Deserialize, Serialize};

/// Structured-output preference for chat responses.
///
/// Providers translate this semantic hint to their native response-format,
/// JSON-mode, schema, or tool-calling controls when supported. Provider-native
/// output controls that are not represented here should be supplied through
/// [`crate::types::chat::ChatRequest::with_provider_option`].
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Request ordinary text output.
    Text,
    /// Request a JSON object without a caller-supplied schema.
    JsonObject,
    /// Request JSON output constrained by a caller-supplied schema.
    JsonSchema {
        /// Schema contract used to guide the model output.
        json_schema: ResponseFormatJsonSchema,
    },
}

/// JSON Schema response-format contract.
///
/// This is a provider-neutral schema hint. Providers may transform, reject, or
/// partially support individual JSON Schema features according to their native
/// structured-output capabilities.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ResponseFormatJsonSchema {
    /// A description of what the response format is for, used by the model to
    /// determine how to respond in the format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The name of the response format. Must be a-z, A-Z, 0-9, or contain
    /// underscores and dashes, with a maximum length of 64.
    pub name: String,
    /// The schema for the response format, described as a JSON Schema object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    /// Whether to enable strict schema adherence when generating the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}
