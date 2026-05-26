use crate::types::chat::{
    AudioPart, ChatMessage, CustomToolDefinition, FilePart, FunctionTool, ImagePart, SystemMessage,
    TextContent, TextPart, ToolDefinition, UserContent, UserContentPart,
};
use crate::types::shared::{FunctionName, ImageUrl};

impl From<&str> for TextPart {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for TextPart {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for SystemMessage {
    fn from(value: &str) -> Self {
        Self {
            content: TextContent::Text(value.to_string()),
            name: None,
        }
    }
}

impl From<String> for SystemMessage {
    fn from(value: String) -> Self {
        Self {
            content: TextContent::Text(value),
            name: None,
        }
    }
}

impl From<Vec<UserContentPart>> for crate::types::chat::UserMessage {
    fn from(value: Vec<UserContentPart>) -> Self {
        Self {
            content: UserContent::Parts(value),
            name: None,
        }
    }
}

impl From<AudioPart> for Vec<UserContentPart> {
    fn from(value: AudioPart) -> Self {
        vec![value.into()]
    }
}

impl From<FilePart> for Vec<UserContentPart> {
    fn from(value: FilePart) -> Self {
        vec![value.into()]
    }
}

impl From<&str> for FunctionName {
    fn from(value: &str) -> Self {
        Self {
            name: value.to_string(),
        }
    }
}

impl From<String> for FunctionName {
    fn from(value: String) -> Self {
        Self { name: value }
    }
}

impl From<&str> for ImageUrl {
    fn from(value: &str) -> Self {
        Self {
            url: value.to_string(),
            detail: Default::default(),
        }
    }
}

impl From<String> for ImageUrl {
    fn from(value: String) -> Self {
        Self {
            url: value,
            detail: Default::default(),
        }
    }
}

impl From<ImageUrl> for ImagePart {
    fn from(value: ImageUrl) -> Self {
        Self {
            image: value,
            cache_control: None,
        }
    }
}

impl From<FunctionTool> for Vec<ToolDefinition> {
    fn from(value: FunctionTool) -> Self {
        vec![ToolDefinition::Function(value)]
    }
}

impl From<CustomToolDefinition> for Vec<ToolDefinition> {
    fn from(value: CustomToolDefinition) -> Self {
        vec![ToolDefinition::Custom(value)]
    }
}

impl From<crate::types::chat::UserMessage> for Vec<ChatMessage> {
    fn from(value: crate::types::chat::UserMessage) -> Self {
        vec![value.into()]
    }
}

impl From<SystemMessage> for Vec<ChatMessage> {
    fn from(value: SystemMessage) -> Self {
        vec![value.into()]
    }
}
