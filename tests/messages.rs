use sigma::types::chat::{
    AssistantContent, AssistantMessage, ChatMessage, DeveloperMessage, Role, SystemMessage,
    TextContent, TextPart, ToolContent, ToolMessage, UserContent, UserContentPart, UserMessage,
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
fn user_message_with_text_content() {
    let m = ChatMessage::User(UserMessage {
        content: UserContent::Text("hi".into()),
        name: None,
    });

    let s = serde_json::to_string(&m).unwrap();

    assert_eq!(s, r#"{"role":"user","content":{"text":"hi"}}"#);
}

#[test]
fn user_message_with_parts_content() {
    let m = UserMessage {
        content: UserContent::Parts(vec![UserContentPart::Text(TextPart {
            text: "hi".into(),
            cache_control: None,
        })]),
        name: None,
    };

    let s = serde_json::to_string(&m).unwrap();

    assert_eq!(s, r#"{"content":{"parts":[{"type":"text","text":"hi"}]}}"#);
}

#[test]
fn system_developer_tool_round_trip() {
    let s = SystemMessage {
        content: TextContent::Text("sys".into()),
        name: None,
    };
    assert_eq!(
        serde_json::to_string(&s).unwrap(),
        r#"{"content":{"text":"sys"}}"#
    );

    let d = DeveloperMessage {
        content: TextContent::Text("dev".into()),
        name: None,
    };
    assert_eq!(
        serde_json::to_string(&d).unwrap(),
        r#"{"content":{"text":"dev"}}"#
    );

    let t = ToolMessage {
        content: ToolContent::Text("ok".into()),
        tool_call_id: "call_1".into(),
    };
    assert_eq!(
        serde_json::to_string(&t).unwrap(),
        r#"{"content":{"text":"ok"},"tool_call_id":"call_1"}"#
    );
}

#[test]
fn assistant_minimal_skips_none() {
    let a = AssistantMessage {
        content: Some(AssistantContent::Text("hi".into())),
        refusal: None,
        name: None,
        audio: None,
        tool_calls: None,
        reasoning: None,
        provider_context: None,
    };

    let s = serde_json::to_string(&a).unwrap();

    assert_eq!(s, r#"{"content":{"text":"hi"}}"#);
}

#[test]
fn message_enum_round_trips_user_and_assistant() {
    let json = r#"[{"role":"user","content":{"text":"hi"}},{"role":"assistant","content":{"text":"hello"}}]"#;

    let v: Vec<ChatMessage> = serde_json::from_str(json).unwrap();

    assert!(matches!(v[0], ChatMessage::User(_)));
    assert!(matches!(v[1], ChatMessage::Assistant(_)));
}
