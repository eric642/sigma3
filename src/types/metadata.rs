use serde::{Deserialize, Serialize};

/// Set of 16 key-value pairs that can be attached to an object.
/// Keys are strings ≤ 64 chars; values are strings ≤ 512 chars.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(transparent)]
pub struct Metadata(serde_json::Value);

impl From<serde_json::Value> for Metadata {
    fn from(value: serde_json::Value) -> Self {
        Self(value)
    }
}
