use sigma::types::chat::{
    ChatChoice, ChatResponse, ChatResponseMessage, ChoiceLogprobs, FinishReason, Role, UrlCitation,
    Usage,
};

#[test]
fn finish_reason_snake_case() {
    let s = serde_json::to_string(&FinishReason::ToolCalls).unwrap();
    assert_eq!(s, r#""tool_calls""#);
}

#[test]
fn url_citation_round_trip() {
    let v = UrlCitation {
        end_index: 10,
        start_index: 0,
        title: "T".into(),
        url: "u".into(),
    };

    let s = serde_json::to_string(&v).unwrap();
    let back: UrlCitation = serde_json::from_str(&s).unwrap();

    assert_eq!(v, back);
}

#[test]
fn response_message_minimal() {
    let m = ChatResponseMessage {
        content: Some("hi".into()),
        reasoning: None,
        refusal: None,
        tool_calls: None,
        annotations: None,
        role: Role::Assistant,
        audio: None,
        provider_context: None,
    };

    let s = serde_json::to_string(&m).unwrap();

    assert_eq!(s, r#"{"content":"hi","role":"assistant"}"#);
}

#[test]
fn chat_response_round_trip() {
    let json = r#"{"id":"x","choices":[{"index":0,"message":{"content":"hi","role":"assistant"}}],"created":1,"model":"gpt-4o","object":"chat.completion","usage":null}"#;

    let r: ChatResponse = serde_json::from_str(json).unwrap();

    assert_eq!(r.id, "x");
    assert_eq!(r.choices.len(), 1);
}

#[test]
fn usage_default_serializes_required_token_counts() {
    let u = Usage::default();

    let s = serde_json::to_string(&u).unwrap();

    assert!(s.contains(r#""prompt_tokens":0"#));
}

#[test]
fn chat_choice_logprobs_optional() {
    let v: ChoiceLogprobs = serde_json::from_str(r#"{"content":null,"refusal":null}"#).unwrap();

    assert!(v.content.is_none());
    assert!(v.refusal.is_none());
}

#[test]
fn choice_minimal() {
    let json = r#"{"index":0,"message":{"content":"hi","role":"assistant"}}"#;

    let _: ChatChoice = serde_json::from_str(json).unwrap();
}
