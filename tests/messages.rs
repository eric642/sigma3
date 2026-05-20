use sigma::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestDeveloperMessage, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, Role,
};

#[test]
fn role_serializes_lowercase() {
    let s = serde_json::to_string(&Role::User).unwrap();
    assert_eq!(s, r#""user""#);
}

#[test]
fn role_default_is_user() {
    assert_eq!(Role::default(), Role::User);
}

#[test]
fn user_message_with_string_content() {
    let m = ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Text("hi".into()),
        name: None,
    });
    let s = serde_json::to_string(&m).unwrap();
    assert_eq!(s, r#"{"role":"user","content":"hi"}"#);
}

#[test]
fn user_message_with_array_content() {
    let m = ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Array(vec![
            ChatCompletionRequestUserMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText { text: "hi".into() },
            ),
        ]),
        name: None,
    };
    let s = serde_json::to_string(&m).unwrap();
    assert_eq!(s, r#"{"content":[{"type":"text","text":"hi"}]}"#);
}

#[test]
fn system_developer_tool_round_trip() {
    let s = ChatCompletionRequestSystemMessage {
        content: ChatCompletionRequestSystemMessageContent::Text("sys".into()),
        name: None,
    };
    assert_eq!(serde_json::to_string(&s).unwrap(), r#"{"content":"sys"}"#);

    let d = ChatCompletionRequestDeveloperMessage {
        content: ChatCompletionRequestDeveloperMessageContent::Text("dev".into()),
        name: None,
    };
    assert_eq!(serde_json::to_string(&d).unwrap(), r#"{"content":"dev"}"#);

    let t = ChatCompletionRequestToolMessage {
        content: ChatCompletionRequestToolMessageContent::Text("ok".into()),
        tool_call_id: "call_1".into(),
    };
    assert_eq!(
        serde_json::to_string(&t).unwrap(),
        r#"{"content":"ok","tool_call_id":"call_1"}"#
    );
}

#[test]
fn assistant_minimal_skips_none() {
    let a = ChatCompletionRequestAssistantMessage {
        content: Some(ChatCompletionRequestAssistantMessageContent::Text(
            "hi".into(),
        )),
        refusal: None,
        name: None,
        audio: None,
        tool_calls: None,
    };
    let s = serde_json::to_string(&a).unwrap();
    assert_eq!(s, r#"{"content":"hi"}"#);
}

#[test]
fn message_enum_round_trips_user_and_assistant() {
    let json = r#"[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]"#;
    let v: Vec<ChatCompletionRequestMessage> = serde_json::from_str(json).unwrap();
    assert!(matches!(v[0], ChatCompletionRequestMessage::User(_)));
    assert!(matches!(v[1], ChatCompletionRequestMessage::Assistant(_)));
}
