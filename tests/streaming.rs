use sigma::types::chat::{
    ChatChoiceStream, ChatCompletionMessageToolCallChunk, ChatCompletionStreamResponseDelta,
    CreateChatCompletionStreamResponse, FunctionCallStream, FunctionType, Role,
};

#[test]
fn function_type_is_lowercase() {
    let s = serde_json::to_string(&FunctionType::Function).unwrap();
    assert_eq!(s, r#""function""#);
}

#[test]
fn delta_round_trip() {
    let d = ChatCompletionStreamResponseDelta {
        content: Some("hi".into()),
        reasoning_content: None,
        tool_calls: None,
        role: Some(Role::Assistant),
        refusal: None,
        provider_specific_fields: None,
    };
    let s = serde_json::to_string(&d).unwrap();
    let back: ChatCompletionStreamResponseDelta = serde_json::from_str(&s).unwrap();
    assert_eq!(d, back);
}

#[test]
fn tool_call_chunk_round_trip() {
    let c = ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call_1".into()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("f".into()),
            arguments: Some("{}".into()),
        }),
        provider_specific_fields: None,
    };
    let s = serde_json::to_string(&c).unwrap();
    let back: ChatCompletionMessageToolCallChunk = serde_json::from_str(&s).unwrap();
    assert_eq!(c, back);
}

#[test]
fn stream_response_round_trip() {
    let json = r#"{"id":"x","choices":[],"created":1,"model":"gpt-4o","service_tier":null,"object":"chat.completion.chunk","usage":null}"#;
    let r: CreateChatCompletionStreamResponse = serde_json::from_str(json).unwrap();
    assert_eq!(r.id, "x");
    assert!(r.choices.is_empty());
}

#[test]
fn choice_stream_minimal() {
    let cs = ChatChoiceStream {
        index: 0,
        delta: ChatCompletionStreamResponseDelta {
            content: Some("a".into()),
            reasoning_content: None,
            tool_calls: None,
            role: None,
            refusal: None,
            provider_specific_fields: None,
        },
        finish_reason: None,
        logprobs: None,
    };
    let _ = serde_json::to_string(&cs).unwrap();
}
