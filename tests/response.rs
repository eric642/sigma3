use sigma::types::Metadata;
use sigma::types::chat::{
    ChatChoice, ChatChoiceLogprobs, ChatCompletionDeleted, ChatCompletionList,
    ChatCompletionResponseMessage, CompletionFinishReason, CompletionUsage,
    CreateChatCompletionResponse, FinishReason, Logprobs, Role, UpdateChatCompletionRequestArgs,
    UrlCitation,
};

#[test]
fn finish_reason_snake_case() {
    let s = serde_json::to_string(&FinishReason::ToolCalls).unwrap();
    assert_eq!(s, r#""tool_calls""#);
}

#[test]
fn completion_finish_reason_snake_case() {
    let s = serde_json::to_string(&CompletionFinishReason::ContentFilter).unwrap();
    assert_eq!(s, r#""content_filter""#);
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
    let m = ChatCompletionResponseMessage {
        content: Some("hi".into()),
        refusal: None,
        tool_calls: None,
        annotations: None,
        role: Role::Assistant,
        audio: None,
        thinking_blocks: None,
        reasoning_content: None,
        provider_specific_fields: None,
    };
    let s = serde_json::to_string(&m).unwrap();
    assert_eq!(s, r#"{"content":"hi","role":"assistant"}"#);
}

#[test]
fn create_response_round_trip() {
    let json = r#"{"id":"x","choices":[{"index":0,"message":{"content":"hi","role":"assistant"}}],"created":1,"model":"gpt-4o","object":"chat.completion","usage":null}"#;
    let r: CreateChatCompletionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(r.id, "x");
    assert_eq!(r.choices.len(), 1);
}

#[test]
fn completion_list_round_trip() {
    let json = r#"{"object":"list","data":[],"first_id":null,"last_id":null,"has_more":false}"#;
    let l: ChatCompletionList = serde_json::from_str(json).unwrap();
    assert!(!l.has_more);
}

#[test]
fn deleted_round_trip() {
    let v = ChatCompletionDeleted {
        object: "chat.completion".into(),
        id: "x".into(),
        deleted: true,
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: ChatCompletionDeleted = serde_json::from_str(&s).unwrap();
    assert_eq!(v, back);
}

#[test]
fn update_chat_completion_request_builder() {
    let req = UpdateChatCompletionRequestArgs::default()
        .metadata(Metadata::from(serde_json::json!({"k": "v"})))
        .build()
        .unwrap();
    let s = serde_json::to_string(&req).unwrap();
    assert_eq!(s, r#"{"metadata":{"k":"v"}}"#);
}

#[test]
fn legacy_logprobs_struct() {
    let v = Logprobs {
        tokens: vec!["a".into()],
        token_logprobs: vec![Some(-0.1)],
        top_logprobs: vec![],
        text_offset: vec![0],
    };
    let s = serde_json::to_string(&v).unwrap();
    let back: Logprobs = serde_json::from_str(&s).unwrap();
    assert_eq!(v, back);
}

#[test]
fn completion_usage_default() {
    let u = CompletionUsage::default();
    let s = serde_json::to_string(&u).unwrap();
    assert!(s.contains(r#""prompt_tokens":0"#));
}

#[test]
fn chat_choice_logprobs_optional() {
    let v: ChatChoiceLogprobs = serde_json::from_str(r#"{"content":null,"refusal":null}"#).unwrap();
    assert!(v.content.is_none());
    assert!(v.refusal.is_none());
}

#[test]
fn choice_minimal() {
    let json = r#"{"index":0,"message":{"content":"hi","role":"assistant"}}"#;
    let _: ChatChoice = serde_json::from_str(json).unwrap();
}
