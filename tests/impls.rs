use sigma::types::chat::{
    AssistantContent, AssistantMessage, ChatMessage, CustomToolDefinition, CustomToolProperties,
    FunctionTool, FunctionToolCall, ImagePart, NamedFunctionToolChoice, Role, SystemMessage,
    TextContent, TextPart, ToolCall, ToolContent, ToolDefinition, UserContent, UserContentPart,
    UserMessage,
};
use sigma::types::shared::{FunctionCall, FunctionName, FunctionObject, ImageUrl};

#[test]
fn role_display() {
    assert_eq!(Role::Assistant.to_string(), "assistant");
    assert_eq!(Role::System.to_string(), "system");
    assert_eq!(Role::Tool.to_string(), "tool");
    assert_eq!(Role::User.to_string(), "user");
}

#[test]
fn user_message_from_str_and_string() {
    let m: UserMessage = "hi".into();
    assert!(matches!(m.content, UserContent::Text(_)));

    let m2: UserMessage = String::from("hi").into();
    assert_eq!(m, m2);
}

#[test]
fn user_message_into_chat_message() {
    let m: ChatMessage = UserMessage::from("hi").into();

    assert!(matches!(m, ChatMessage::User(_)));
}

#[test]
fn assistant_message_from_str() {
    let m: AssistantMessage = "hi".into();

    assert!(matches!(m.content, Some(AssistantContent::Text(_))));
}

#[test]
fn user_content_from_parts_array() {
    let parts: Vec<UserContentPart> = vec![TextPart::new("hi").into()];

    let c: UserContent = parts.into();

    assert!(matches!(c, UserContent::Parts(_)));
}

#[test]
fn user_part_from_text_and_image() {
    let _: UserContentPart = TextPart::new("hi").into();
    let _: UserContentPart = ImagePart {
        image: ImageUrl {
            url: "u".into(),
            detail: None,
        },
        cache_control: None,
    }
    .into();
}

#[test]
fn text_part_from_str() {
    let t: TextPart = "hi".into();

    assert_eq!(t.text, "hi");
}

#[test]
fn image_url_from_str_uses_default_detail() {
    let u: ImageUrl = "https://x.test".into();

    assert_eq!(u.url, "https://x.test");
    assert_eq!(u.detail, None);
}

#[test]
fn image_url_into_image_part() {
    let _: ImagePart = ImageUrl {
        url: "u".into(),
        detail: None,
    }
    .into();
}

#[test]
fn function_name_and_named_choice_from_str() {
    let n: FunctionName = "f".into();
    assert_eq!(n.name, "f");

    let c: NamedFunctionToolChoice = "f".into();
    assert_eq!(c.function.name, "f");
}

#[test]
fn vec_chat_message_from_individual_messages() {
    let v: Vec<ChatMessage> = SystemMessage::from("sys").into();
    assert_eq!(v.len(), 1);

    let v: Vec<ChatMessage> = UserMessage::from("hi").into();
    assert_eq!(v.len(), 1);
}

#[test]
fn tool_call_into_message_tool_calls() {
    let t = FunctionToolCall {
        id: "x".into(),
        function: FunctionCall {
            name: "f".into(),
            arguments: "{}".into(),
        },
        reasoning: None,
    };

    let _: ToolCall = t.into();
}

#[test]
fn tool_into_vec_chat_tools() {
    let t = FunctionTool {
        function: FunctionObject {
            name: "f".into(),
            description: None,
            parameters: None,
            strict: None,
        },
    };
    let v: Vec<ToolDefinition> = t.into();
    assert_eq!(v.len(), 1);

    let c = CustomToolDefinition {
        custom: CustomToolProperties::default(),
    };
    let v: Vec<ToolDefinition> = c.into();
    assert_eq!(v.len(), 1);
}

#[test]
fn defaults_for_message_contents() {
    let _ = UserContent::default();
    let _ = TextContent::default();
    let _ = ToolContent::default();
}
