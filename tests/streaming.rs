use sigma::types::chat::{
    AssistantDelta, ChatStreamChoice, ChatStreamChunk, FunctionCallDelta, Role, ToolCallDelta,
    ToolCallKind,
};

#[test]
fn function_type_is_lowercase() {
    let s = serde_json::to_string(&ToolCallKind::Function).unwrap();
    assert_eq!(s, r#""function""#);
}

#[test]
fn delta_round_trip() {
    let d = AssistantDelta {
        content: Some("hi".into()),
        reasoning: None,
        tool_calls: None,
        role: Some(Role::Assistant),
        refusal: None,
    };

    let s = serde_json::to_string(&d).unwrap();
    let back: AssistantDelta = serde_json::from_str(&s).unwrap();

    assert_eq!(d, back);
}

#[test]
fn tool_call_chunk_round_trip() {
    let c = ToolCallDelta {
        index: 0,
        id: Some("call_1".into()),
        r#type: Some(ToolCallKind::Function),
        function: Some(FunctionCallDelta {
            name: Some("f".into()),
            arguments: Some("{}".into()),
        }),
        reasoning: None,
    };

    let s = serde_json::to_string(&c).unwrap();
    let back: ToolCallDelta = serde_json::from_str(&s).unwrap();

    assert_eq!(c, back);
}

#[test]
fn stream_response_round_trip() {
    let json = r#"{"id":"x","choices":[],"created":1,"model":"gpt-4o","service_tier":null,"object":"chat.completion.chunk","usage":null}"#;

    let r: ChatStreamChunk = serde_json::from_str(json).unwrap();

    assert_eq!(r.id, "x");
    assert!(r.choices.is_empty());
}

#[test]
fn choice_stream_minimal() {
    let cs = ChatStreamChoice {
        index: 0,
        delta: AssistantDelta {
            content: Some("a".into()),
            reasoning: None,
            tool_calls: None,
            role: None,
            refusal: None,
        },
        finish_reason: None,
        logprobs: None,
    };

    let _ = serde_json::to_string(&cs).unwrap();
}
