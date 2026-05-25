use sigma::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls, ChatCompletionNamedToolChoice,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestDeveloperMessage, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionTool, ChatCompletionTools,
    CustomToolChatCompletions, CustomToolProperties, Role,
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
    let m: ChatCompletionRequestUserMessage = "hi".into();
    assert!(matches!(
        m.content,
        ChatCompletionRequestUserMessageContent::Text(_)
    ));
    let m2: ChatCompletionRequestUserMessage = String::from("hi").into();
    assert_eq!(m, m2);
}

#[test]
fn user_message_into_request_message() {
    let m: ChatCompletionRequestMessage = ChatCompletionRequestUserMessage::from("hi").into();
    assert!(matches!(m, ChatCompletionRequestMessage::User(_)));
}

#[test]
fn assistant_message_from_str() {
    let m: ChatCompletionRequestAssistantMessage = "hi".into();
    assert!(matches!(
        m.content,
        Some(ChatCompletionRequestAssistantMessageContent::Text(_))
    ));
}

#[test]
fn user_content_from_parts_array() {
    let parts: Vec<ChatCompletionRequestUserMessageContentPart> =
        vec![ChatCompletionRequestUserMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: "hi".into() },
        )];
    let c: ChatCompletionRequestUserMessageContent = parts.into();
    assert!(matches!(
        c,
        ChatCompletionRequestUserMessageContent::Array(_)
    ));
}

#[test]
fn user_part_from_text_image_audio() {
    let _: ChatCompletionRequestUserMessageContentPart =
        ChatCompletionRequestMessageContentPartText { text: "hi".into() }.into();
    let _: ChatCompletionRequestUserMessageContentPart =
        ChatCompletionRequestMessageContentPartImage {
            image_url: ImageUrl {
                url: "u".into(),
                detail: None,
            },
        }
        .into();
}

#[test]
fn text_part_from_str() {
    let t: ChatCompletionRequestMessageContentPartText = "hi".into();
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
    let _: ChatCompletionRequestMessageContentPartImage = ImageUrl {
        url: "u".into(),
        detail: None,
    }
    .into();
}

#[test]
fn function_name_and_named_choice_from_str() {
    let n: FunctionName = "f".into();
    assert_eq!(n.name, "f");
    let c: ChatCompletionNamedToolChoice = "f".into();
    assert_eq!(c.function.name, "f");
}

#[test]
fn vec_request_message_from_individual_messages() {
    let v: Vec<ChatCompletionRequestMessage> =
        ChatCompletionRequestSystemMessage::from("sys").into();
    assert_eq!(v.len(), 1);
    let v: Vec<ChatCompletionRequestMessage> = ChatCompletionRequestUserMessage::from("hi").into();
    assert_eq!(v.len(), 1);
    let v: Vec<ChatCompletionRequestMessage> =
        ChatCompletionRequestDeveloperMessage::from("dev").into();
    assert_eq!(v.len(), 1);
    let v: Vec<ChatCompletionRequestMessage> =
        ChatCompletionRequestAssistantMessage::from("a").into();
    assert_eq!(v.len(), 1);
}

#[test]
fn tool_call_into_message_tool_calls() {
    let t = ChatCompletionMessageToolCall {
        id: "x".into(),
        function: FunctionCall {
            name: "f".into(),
            arguments: "{}".into(),
        },
        provider_specific_fields: None,
    };
    let _: ChatCompletionMessageToolCalls = t.into();
}

#[test]
fn tool_into_vec_chat_completion_tools() {
    let t = ChatCompletionTool {
        function: FunctionObject {
            name: "f".into(),
            description: None,
            parameters: None,
            strict: None,
        },
    };
    let v: Vec<ChatCompletionTools> = t.into();
    assert_eq!(v.len(), 1);

    let c = CustomToolChatCompletions {
        custom: CustomToolProperties::default(),
    };
    let v: Vec<ChatCompletionTools> = c.into();
    assert_eq!(v.len(), 1);
}

#[test]
fn defaults_for_message_contents() {
    let _ = ChatCompletionRequestUserMessageContent::default();
    let _ = ChatCompletionRequestSystemMessageContent::default();
    let _ = ChatCompletionRequestDeveloperMessageContent::default();
    let _ = ChatCompletionRequestToolMessageContent::default();
}
