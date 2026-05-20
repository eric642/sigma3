use sigma::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, Prompt, StopConfiguration,
};

#[test]
fn prompt_untagged_string() {
    let s = serde_json::to_string(&Prompt::String("hi".into())).unwrap();
    assert_eq!(s, r#""hi""#);
}

#[test]
fn prompt_untagged_string_array() {
    let s = serde_json::to_string(&Prompt::StringArray(vec!["a".into(), "b".into()])).unwrap();
    assert_eq!(s, r#"["a","b"]"#);
}

#[test]
fn stop_configuration_untagged() {
    let s = serde_json::to_string(&StopConfiguration::String("\n".into())).unwrap();
    assert_eq!(s, r#""\n""#);
    let s = serde_json::to_string(&StopConfiguration::StringArray(vec!["a".into()])).unwrap();
    assert_eq!(s, r#"["a"]"#);
}

#[test]
fn create_request_minimal_skips_none() {
    let user = ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Text("hi".into()),
        name: None,
    };
    let req = CreateChatCompletionRequestArgs::default()
        .messages(vec![ChatCompletionRequestMessage::User(user)])
        .model("gpt-4o")
        .build()
        .unwrap();
    let s = serde_json::to_string(&req).unwrap();
    assert_eq!(
        s,
        r#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o"}"#
    );
}

#[test]
fn create_request_round_trip() {
    let json =
        r#"{"messages":[{"role":"user","content":"hi"}],"model":"gpt-4o","temperature":0.7}"#;
    let req: CreateChatCompletionRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.model, "gpt-4o");
    assert_eq!(req.temperature, Some(0.7));
}
